from argparse import Namespace
from math import log2
from typing import Optional, Dict, Any, Union, TYPE_CHECKING, Tuple
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
from .hash_grid_torch import HashEmbedder
from ..transform import RigidTransform, ax_transform_points, mat_transform_points
from ..utils import resolution2sigma
from .net_v3 import volumeNet
from .data import PointDataset
from .utils import *
from pytorch3d.structures import Volumes, Pointclouds
from pytorch3d.ops import add_pointclouds_to_volumes
import math
import numpy as np
from .functional import trilinear_devoxelize
USE_TORCH = False

if not USE_TORCH:
	try:
		import tinycudann as tcnn
	except:
		logging.warning("Fail to load tinycudann. Will use pytorch implementation.")
		USE_TORCH = True


# key for loss/regularization
D_LOSS = "MSE"
S_LOSS = "logVar"
DS_LOSS = "MSE+logVar"
B_REG = "biasReg"
T_REG = "transReg"
I_REG = "imageReg"
D_REG = "deformReg"

def select_idx(*args, idx=None):
	if idx is None:
		return args
	return tuple(arg[idx] for arg in args)

def check_nan(tensor_dict):
	is_nan = False
	nan_list = []
	for k, v in tensor_dict.items():
		if v.isnan().any() or v.isinf().any():
			is_nan = True
			nan_list.append(k)

	if is_nan:
		logging.warning("Nan in ", nan_list, " Please debug")
		breakpoint()


def devoxelize_int_mask(volume: torch.Tensor, coords: torch.Tensor):
	"""
	Args:
		volume: (h, w, l, feature_dim)
		coords: (npoints, 3)
	"""
	assert len(coords.shape) == 2 and len(volume.shape) == 4, \
		 "shape should be mask: (npoints, 3); volume: (h, w, l, feature_dim)"

	if (coords.amax(0) > volume.shape[0]).any() or (coords.amin(0) < 0).any():
			logging.warning("mask out of bound!!! You must debug now")
			breakpoint()
			
	coords = coords.int()
	return volume[coords[:, 0], coords[:, 1], coords[:, 2]]


@torch.autocast("cuda", enabled=False)
def devoxelize_trilinear(volume: torch.Tensor, coords: torch.Tensor):
	"""
	Args:
		volume: (h, w, l, feature_dim)
		coords: (npoints, 3)
	"""
	assert len(coords.shape) == 2 and len(volume.shape) == 4, \
		 "shape should be mask: (npoints, 3); volume: (h, w, l, feature_dim)"
	# Check out of bound
	max_coord = torch.tensor(volume.shape[:-1], device=coords.device, dtype=coords.dtype) - 1
	min_coord = torch.zeros(3, device=coords.device, dtype=coords.dtype)
	num_outside = ((coords < 0).any(1) | (coords >= max_coord).any(1)).sum().item()  
	if num_outside > 0:
		print(f"{num_outside} coordinates are mapped outside the volume and thus clipped")
	coords.clamp_(min_coord, max_coord)

	coords = coords.float() 
	volume = volume.float()
	resolution = volume.shape[1]
	volume = trilinear_devoxelize(volume.permute(3, 0, 1, 2).unsqueeze(0), coords.unsqueeze(0).permute(0, 2, 1), resolution)
	volume = volume.squeeze(0).permute(1, 0) 
	return volume


def build_encoding(**config):
	if USE_TORCH:
		encoding = HashEmbedder(**config)
	else:
		n_input_dims = config.pop("n_input_dims")
		dtype = config.pop("dtype")
		try:
			encoding = tcnn.Encoding(
				n_input_dims=n_input_dims, encoding_config=config, dtype=dtype
			)
		except RuntimeError as e:
			if "TCNN was not compiled with half-precision support" in str(e):
				logging.error(
					"TCNN was not compiled with half-precision support! "
					"Try using --single-precision in the nesvor command! "
				)
			raise e
	return encoding


def build_network(**config):
	dtype = config.pop("dtype")
	assert dtype == torch.float16 or dtype == torch.float32
	if dtype == torch.float16 and not USE_TORCH:
		return tcnn.Network(
			n_input_dims=config["n_input_dims"],
			n_output_dims=config["n_output_dims"],
			network_config={
				"otype": "CutlassMLP",
				"activation": config["activation"],
				"output_activation": config["output_activation"],
				"n_neurons": config["n_neurons"],
				"n_hidden_layers": config["n_hidden_layers"],
			},
		)
	else:
		activation = (
			None
			if config["activation"] == "None"
			else getattr(nn, config["activation"])
		)
		output_activation = (
			None
			if config["output_activation"] == "None"
			else getattr(nn, config["output_activation"])
		)
		models = []
		if config["n_hidden_layers"] > 0:
			models.append(nn.Linear(config["n_input_dims"], config["n_neurons"]))
			for _ in range(config["n_hidden_layers"] - 1):
				if activation is not None:
					models.append(activation())
				models.append(nn.Linear(config["n_neurons"], config["n_neurons"]))
			if activation is not None:
				models.append(activation())
			models.append(nn.Linear(config["n_neurons"], config["n_output_dims"]))
		else:
			models.append(nn.Linear(config["n_input_dims"], config["n_output_dims"]))
		if output_activation is not None:
			models.append(output_activation())
		return nn.Sequential(*models)


def compute_resolution_nlevel(
	bounding_box: torch.Tensor,
	coarsest_resolution: float,
	finest_resolution: float,
	level_scale: float,
	spatial_scaling: float,
) -> Tuple[int, int]:
	base_resolution = (
		(
			(bounding_box[1] - bounding_box[0]).max()
			* spatial_scaling
			/ coarsest_resolution
		)
		.ceil()
		.int()
		.item()
	)
	n_levels = (
		(
			torch.log2(
				(bounding_box[1] - bounding_box[0]).max()
				* spatial_scaling
				/ finest_resolution
				/ base_resolution
			)
			/ log2(level_scale)
			+ 1
		)
		.ceil()
		.int()
		.item()
	)
	return int(base_resolution), int(n_levels)


class INR(nn.Module):
	def __init__(
		self, bounding_box: torch.Tensor,
		args: Namespace,
		spatial_scaling: float = 1.0,
		resolutions = None,
		original_shape: torch.Tensor = None,
		grid_upsample_rate: int = 1,
	) -> None:
		"""
		Takes in discrete 3D coordinates within the bounding box 
		"""
		super().__init__()
		if TYPE_CHECKING:
			self.bounding_box: torch.Tensor
		self.register_buffer("bounding_box", bounding_box)
		self.xyz_span_this = bounding_box[1] - bounding_box[0]
		self.resolutions = resolutions
		self.args = args
		self.original_shape = original_shape # shape of the gt volume
		self.trilinear_devox = args.trilinear_devox
		# hash grid encoding
		base_resolution, n_levels = compute_resolution_nlevel(
			self.bounding_box,
			args.coarsest_resolution,
			args.finest_resolution,
			args.level_scale,
			spatial_scaling,
		)
		if self.args.hash_path:
			self.encoding = torch.load(self.args.hash_path)
			self.encoding.eval()
			freeze_params(self.encoding.parameters())
		else:
			self.encoding = build_encoding(
				n_input_dims=3,
				otype="HashGrid",
				n_levels=n_levels,
				n_features_per_level=args.n_features_per_level,
				log2_hashmap_size=args.log2_hashmap_size,
				base_resolution=base_resolution,    # @wenxuan
				per_level_scale=args.level_scale,   # the resolution/grid size multiplier between levels, by which coords are multiplied 
													# Low res -> high res, low freq -> high freq
				dtype=args.dtype,
			)

		self.o_inr = args.o_inr
		self.upsample_rate = grid_upsample_rate
		# density net
		if self.o_inr:   
			
			outchannel = 3 if self.args.add_ch else 1 + args.n_features_z
			self.density_net = volumeNet(
				None,
				None,
				inchannel=n_levels * args.n_features_per_level,
				outchannel=outchannel,
				learnable_wave=None,
				transform=None,
				mode=None,
				use_ckconv=self.args.ckconv,
				upsample_rate=grid_upsample_rate,
			)
			self.forward = self._volume_forward
			self.num_patches = 5
			self.patch_idx = torch.randint(self.num_patches, size=(1, )).item()

		else:
			self.density_net = build_network(
				n_input_dims=n_levels * args.n_features_per_level,
				n_output_dims=1 + args.n_features_z,
				activation="ReLU",
				output_activation="None",
				n_neurons=args.width,
				n_hidden_layers=args.depth,
				dtype=torch.float32 if args.img_reg_autodiff else args.dtype,
			) 
		
		# logging
		logging.debug(
			"hyperparameters for hash grid encoding: "
			+ "lowest_grid_size=%d, highest_grid_size=%d, scale=%1.2f, n_levels=%d",
			base_resolution,
			int(base_resolution * args.level_scale ** (n_levels - 1)),
			args.level_scale,
			n_levels,
		)
		logging.debug(
			"bounding box for reconstruction (mm): "
			+ "x=[%f, %f], y=[%f, %f], z=[%f, %f]",
			self.bounding_box[0, 0],
			self.bounding_box[1, 0],
			self.bounding_box[0, 1],
			self.bounding_box[1, 1],
			self.bounding_box[0, 2],
			self.bounding_box[1, 2],
		)

	def get_voxel_grid(self, pad_len, new_res, feature_dim=1) -> Volumes:
		"""
		Simulate the original voxel grid (coordinates) from which stacks are extracted
		Args:
			new_res: resolution of the new voxel grid 
		"""
		xyz_span = self.original_shape
		xyz_span = (xyz_span * new_res).ceil().int()
		device = self.resolutions.device

		new_size = (xyz_span + pad_len * 2).int().tolist()
		# (minibatch, densities_dim, height, width, depth)
		densities = torch.zeros([1, 1] + new_size, dtype=torch.float32, device=device)
		features = torch.zeros([1, feature_dim] + new_size, dtype=torch.float32, device=device)
		volume_in = Volumes(densities, features)
		return volume_in
	
 
	def range(self, xyz: torch.Tensor):
		"""
		Args:
			xyz: (npoints, 3) or (1, npoints, 3)
		"""
		xyz = xyz.squeeze()
		xyz_min = xyz.amin(0)
		xyz_max = xyz.amax(0)

		return xyz_max - xyz_min


	def points2new_grid(
					self,
		 			xyz: torch.Tensor,
					feature_dim: int = 1,
					zero_one: bool = False,
					new_resolution=0.7,
					round_coords=False
					) -> Tuple[torch.Tensor, Volumes]:
		"""
		Convert slice coordinates to target grid coordinates and create the voxel grid.
		Args:
			xyz: (npoints, 3) transformed coordinates for INR training.
			values: (npoints, ) pixel intensities. Will average neighbors if provided.
		"""
		self.final_pad_len = self.resolutions.max() 
		device = self.resolutions.device
		
		# Get voxel grid
		vol_in = self.get_voxel_grid(self.final_pad_len, new_resolution, feature_dim)
		self.xyz_span_dest = vol_in.densities().squeeze().shape	# (h, w, l)
		# Verify 3d shape
		assert len(self.xyz_span_dest) == 3, "volume shape should be 3D" 

		# Apply scaling. Coords should be non-negative for the use of hash grid
		self.xyz_span_dest = torch.tensor(self.xyz_span_dest, device=device) 
		xyz = (xyz - self.bounding_box[0]) / self.xyz_span_this 
		xyz = xyz * self.xyz_span_dest
		
		if not zero_one:
			xyz = xyz - self.xyz_span_dest / 2
			
		# Count out of bound points
		num_outside = 0

		if zero_one:
			num_outside += (xyz < 0).any(1).sum().item()
			num_outside += (xyz >= self.xyz_span_dest).any(1).sum().item()
		else:
			xyz_bound = (self.xyz_span_dest - 1) / 2
			num_outside += (xyz >= xyz_bound ).any(1).sum().item()
			num_outside += (xyz <= -xyz_bound).any(1).sum().item()
		
		if num_outside > 0:
			logging.warning(f"{num_outside} coordinates are mapped outside the volume")
		
		if round_coords:
			xyz = xyz.round().int()

		return xyz, vol_in
		

	def voxelize(
			self,
			xyz: torch.Tensor,
			features: torch.tensor,
			new_resolution: float,
			round_coords: bool,
			):
		
		xyz, vol_in = self.points2new_grid(
					xyz,
					feature_dim=features.shape[-1],
					zero_one=False,
					new_resolution=new_resolution,
					round_coords=round_coords
					)
		cloud = Pointclouds(xyz[None].float(), features = features[None])
		# Interpolate onto 8 surrounding vertices
		mode = "trilinear" 
		vol_in = add_pointclouds_to_volumes(
			cloud, vol_in, mode=mode, 
		)
		return xyz, vol_in


	@torch.autocast("cuda", enabled=False)
	def point2voxel(
			self,
			xyz: torch.Tensor,
			features: torch.Tensor,
			new_res=0.7,
			patchify=False,
			return_mask=False,
			trilinear_devox=True
			) \
		-> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Map final (transformed) 3D coordinates and corresponding features to voxels.
		
		Args: 
			xyz: transformed coordinates for INR training. (npoints, 3) or (npoints, n_PSF, 3)
			features: features for INR training. (npoints, feature_dim) or (npoints, n_PSF, feature_dim)
			new_res: grid size as a fraction of the original size
			patchify: whether to split the volume into patches and randomly sample one patch
		"""
		features = features[:, None] if features.ndim == 1 else features # get batch dim
		feature_dim = features.shape[-1] 
		prefix_shape = xyz.shape[:-1]
		xyz = xyz.reshape(-1, 3)
		features = features.reshape(-1, feature_dim)
		
		round_coords = not trilinear_devox
		# Voxelize points and corresponding features 
		xyz, vol_in = self.voxelize(xyz, features, new_res, round_coords)
		features = vol_in.features().squeeze(0).permute(1, 2, 3, 0) #  -> (h, w, l, feature_dim)
		mesh_grid = vol_in.get_coord_grid(world_coordinates=False).squeeze() # (h, w, l, 3)

		# Get GT feature pointcloud from volume 
		mask = (xyz + (vol_in.get_grid_sizes() - 1) / 2)
		# assert torch.allclose(mask, vol_in.local_to_world_coords( (vol_in.world_to_local_coords(xyz.float()) + 1) ).squeeze()), \
		#  "mask transform doesn't align with pytorch3d"
		mask = mask.round().int() if not trilinear_devox else mask

		# Select one patch from the whole volume
		if patchify:
			patch_size = math.ceil(mesh_grid.shape[1] / self.num_patches)
			start = self.patch_idx * patch_size
			end = min(start + patch_size, mesh_grid.shape[1] - 1)
			mask = mask[(mask[:, 2] >= start) & (mask[:, 2] <= end), :] # Take patches along z-axis
			prefix_shape = torch.Size([-1])
			
		# Devoxelization
		if not trilinear_devox:
			features = devoxelize_int_mask(features, mask).reshape(*prefix_shape, feature_dim)
		else:
			features = devoxelize_trilinear(features, mask).reshape(*prefix_shape, feature_dim)
		mesh_grid = (mesh_grid + 1) / 2  # Rescale to [0, 1] as needed by the hash grid
		features = features.reshape(*prefix_shape, feature_dim)
		
		return_list = [mesh_grid]
		return_list += [features]
		if return_mask:
			return_list += [mask.int()]
		
		return return_list

	def reset_patch(self):
		self.patch_idx = torch.randint(self.num_patches, size=(1,)).item()

	@torch.autocast("cuda", cache_enabled=False)
	def _volume_forward(
			self,
			xyz: torch.Tensor,
			values: Optional[torch.Tensor] = None,
			):
		"""
		Args: 
			xyz: (npoints, 3) or (npoints, n_PSF, 3)
			se: (npoints, 16) 
			values: (npoints, ) GT density values 
		"""
		if self.training and values is None:
			raise ValueError(" In training but values not provided to forward, overlapping coordinates won't be hanlded!")
		
		dtype = xyz.dtype
		prefix_shape = xyz.shape[:-1]
		patchify = self.args.patchify and self.training
		new_resolution = self.upsample_rate * 0.7
		
		# Training with ground truth
		if values is not None:	
			vol_in, values, mask = self.point2voxel(
				xyz,
				values,
				patchify = patchify,
				return_mask = True,
				new_res = new_resolution,
				trilinear_devox = self.trilinear_devox
				)
		# Inference
		else:
			mask, vol_in = self.points2new_grid(
				xyz,
				new_resolution = new_resolution,
				zero_one = True,
				round_coords = not self.trilinear_devox
				)
			vol_in = vol_in.get_coord_grid(world_coordinates=False)
			if not self.trilinear_devox:
				mask = mask.round().int()

		grid_shape = vol_in.shape[:-1]
		# To hash grid encodings
		pe = self.encoding(vol_in.reshape(-1, 3)).reshape(grid_shape + (-1,))
		pe = pe.to(dtype=dtype).squeeze()
		pe = pe.unsqueeze(0).permute(0, 4, 1, 2, 3) # (1, encoding_dim, x, y, z)

		# Use additional channels instead of MLPs for log_var, log_bias
		if self.args.add_ch:
			out = self.density_net(pe) # (1, 3, x, y, z)
			out = out.squeeze().permute(1, 2, 3, 0) # (x, y, z, 3)
			if self.trilinear_devox:
				out = devoxelize_trilinear(out, mask)
			else:
				out = devoxelize_int_mask(out, mask)
			
			density = F.softplus(out[:, 0])
			log_var = out[:, 1]
			log_bias = out[:, 2]

			if self.training:
				return density, log_var, log_bias, values
			else:
				return density
			
		z = self.density_net(pe)
		z = z.permute(0, 2, 3, 4, 1).squeeze()
		
		# volume to points
		pe = pe.squeeze(0).permute(1, 2, 3, 0) # (h, w, l, feature_dim)
		if self.trilinear_devox:
			pe = devoxelize_trilinear(pe, mask)
			z = devoxelize_trilinear(z, mask)
		else:
			pe = devoxelize_int_mask(pe, mask)
			z = devoxelize_int_mask(z, mask)
		density = F.softplus(z[:, 0]) 
		
		if not self.args.patchify:
			# Other outputs will be reshaped later to density.shape in Nesvor.forward
			density = density.reshape(prefix_shape)

		if self.training:
			return density, pe, z, values
		
		return density


	def forward(self, x: torch.Tensor, **kwargs):
		prefix_shape = x.shape[:-1] # (batch_size, n_psf_samples, 3)
		dtype = x.dtype
		if not self.args.use_voxel:
			# normalize coordinates to [0, 1]
			x = (x - self.bounding_box[0]) / (self.bounding_box[1] - self.bounding_box[0])
			x = x.view(-1, x.shape[-1])
			pe = self.encoding(x)  
		else:
			# pe, _, _ = self.points2volume(x, return_unique_xyz=False, **kwargs); del x
			raise NotImplementedError("Voxel preprocessing not adapted for INR yet ")
			
		# The density net takes encodings at all levels, 
		# but the other two MLPs split them
		if not self.training:
			pe = pe.to(dtype=dtype)
		z = self.density_net(pe)
		density = F.softplus(z[..., 0].view(prefix_shape))
		if self.training:
			return density, pe, z
		return density


	def sample_batch(
		self,
		xyz: torch.Tensor,
		transformation: Optional[RigidTransform],
		psf_sigma: Union[float, torch.Tensor],
		n_samples: int,
	) -> torch.Tensor:
		xyz = xyz[:, None]

		if n_samples > 1:
			if isinstance(psf_sigma, torch.Tensor):
				psf_sigma = psf_sigma.view(-1, 1, 3)
			xyz_psf = torch.randn(
				xyz.shape[0], n_samples, 3, dtype=xyz.dtype, device=xyz.device
			)
			xyz = xyz + xyz_psf * psf_sigma

		if transformation is not None:
			trans_first = transformation.trans_first
			mat = transformation.matrix(trans_first)
			xyz = mat_transform_points(mat[:, None], xyz, trans_first)
		return xyz

	

class DeformNet(nn.Module):
	def __init__(
		self, bounding_box: torch.Tensor, args: Namespace, spatial_scaling: float = 1.0
	) -> None:
		super().__init__()
		if TYPE_CHECKING:
			self.bounding_box: torch.Tensor
		self.register_buffer("bounding_box", bounding_box)
		# hash grid encoding
		base_resolution, n_levels = compute_resolution_nlevel(
			bounding_box,
			args.coarsest_resolution_deform,
			args.finest_resolution_deform,
			args.level_scale_deform,
			spatial_scaling,
		)
		level_scale = args.level_scale_deform

		self.encoding = build_encoding(
			n_input_dims=3,
			otype="HashGrid",
			n_levels=n_levels,
			n_features_per_level=args.n_features_per_level_deform,
			log2_hashmap_size=args.log2_hashmap_size,
			base_resolution=base_resolution,
			per_level_scale=level_scale,
			dtype=args.dtype,
			interpolation="Smoothstep",
		)
		self.deform_net = build_network(
			n_input_dims=n_levels * args.n_features_per_level_deform
			+ args.n_features_deform,
			n_output_dims=3,
			activation="Tanh",
			output_activation="None",
			n_neurons=args.width,
			n_hidden_layers=2,
			dtype=torch.float32,
		)
		for p in self.deform_net.parameters():
			torch.nn.init.uniform_(p, a=-1e-4, b=1e-4)
		logging.debug(
			"hyperparameters for hash grid encoding (deform net): "
			+ "lowest_grid_size=%d, highest_grid_size=%d, scale=%1.2f, n_levels=%d",
			base_resolution,
			int(base_resolution * level_scale ** (n_levels - 1)),
			level_scale,
			n_levels,
		)

	def forward(self, x: torch.Tensor, e: torch.Tensor):
		x = (x - self.bounding_box[0]) / (self.bounding_box[1] - self.bounding_box[0])
		x_shape = x.shape
		x = x.view(-1, x.shape[-1])
		pe = self.encoding(x)
		inputs = torch.cat((pe, e.reshape(-1, e.shape[-1])), -1)
		outputs = self.deform_net(inputs) + x
		outputs = (
			outputs * (self.bounding_box[1] - self.bounding_box[0])
			+ self.bounding_box[0]
		)
		return outputs.view(x_shape)


class NeSVoR(nn.Module):
	def __init__(
		self,
		transformation: RigidTransform,
		resolution: torch.Tensor,
		v_mean: float,
		bounding_box: torch.Tensor,
		spatial_scaling: float,
		args: Namespace,
		original_shape: torch.Size,
		upsample_rate: int = 1,
	) -> None:
		super().__init__()
		if "cpu" in str(args.device):  # CPU mode
			global USE_TORCH
			USE_TORCH = True
		else:
			# set default GPU for tinycudann
			torch.cuda.set_device(args.device)
		self.spatial_scaling = spatial_scaling
		self.args = args
		self.n_slices = 0
		self.trans_first = True
		self.transformation = transformation
		self.psf_sigma = resolution2sigma(resolution, isotropic=False)
		self.delta = args.delta * v_mean
		self.resolutions = resolution 
		self.original_shape = original_shape

		self.build_network(bounding_box, upsample_rate)
		self.to(args.device)

		# added arguments
		self.o_inr = self.args.o_inr

	@property
	def transformation(self) -> RigidTransform:
		return RigidTransform(self.axisangle.detach(), self.trans_first)

	@transformation.setter
	def transformation(self, value: RigidTransform) -> None:
		if self.n_slices == 0:
			self.n_slices = len(value)
		else:
			assert self.n_slices == len(value)
		axisangle = value.axisangle(self.trans_first)
		if TYPE_CHECKING:
			self.axisangle_init: torch.Tensor
		self.register_buffer("axisangle_init", axisangle.detach().clone())
		if not self.args.no_transformation_optimization:
			self.axisangle = nn.Parameter(axisangle.detach().clone())
		else:
			self.register_buffer("axisangle", axisangle.detach().clone())


	def build_network(self, bounding_box, upsample_rate=1) -> None:
		if self.args.n_features_slice:
			self.slice_embedding = nn.Embedding(
				self.n_slices, self.args.n_features_slice # = 16
			)
		if not self.args.no_slice_scale:
			self.logit_coef = nn.Parameter(
				torch.zeros(self.n_slices, dtype=torch.float32)
			)
		if not self.args.no_slice_variance:
			self.log_var_slice = nn.Parameter(
				torch.zeros(self.n_slices, dtype=torch.float32)
			)
		if self.args.deformable:
			self.deform_embedding = nn.Embedding(
				self.n_slices, self.args.n_features_deform
			)
			self.deform_net = DeformNet(bounding_box, self.args, self.spatial_scaling)
		
		self.inr = INR(bounding_box, self.args, self.spatial_scaling, 
		 			self.resolutions,
					self.original_shape,
					upsample_rate
				)
		# sigma net
		if not self.args.no_pixel_variance:
			self.sigma_net = build_network(
				n_input_dims=self.args.n_features_slice + self.args.n_features_z,
				n_output_dims=1,
				activation="ReLU",
				output_activation="None",
				n_neurons=self.args.width,
				n_hidden_layers=self.args.depth,
				dtype=self.args.dtype,
			)
		# bias net
		if self.args.n_levels_bias:
			self.b_net = build_network(
				n_input_dims=self.args.n_levels_bias * self.args.n_features_per_level
				+ self.args.n_features_slice,
				n_output_dims=1,
				activation="ReLU",
				output_activation="None",
				n_neurons=self.args.width,
				n_hidden_layers=self.args.depth,
				dtype=self.args.dtype,
			)


	def net_forward(
		self,
		x: torch.Tensor,
		se: Optional[torch.Tensor] = None,
		values: Optional[torch.Tensor] = None,
	) -> Dict[str, Any]:

		results = {}

		# NOTE: Main INR forward
		if self.args.add_ch and self.o_inr:
			# Use additional channels to predict all values in eq (3)
			density, log_var, log_bias, values = self.inr(x, values=values)
			results["density"] = density
			results["log_var"] = log_var
			results["log_bias"] = log_bias
			results["values"] = values
			
			check_nan(results)
			return results
		elif self.o_inr:
			density, pe, z, values = self.inr(x, values=values) # outputs are all pointcloud-like values
			_, se = self.inr.point2voxel(x, se, patchify=self.args.patchify)
			results["values"] = values
		else:
			# Original INR
			density, pe, z = self.inr(x, values=values)

		results['density'] = density
		prefix_shape = density.shape

		zs = []
		if se is not None:
			zs.append(se.reshape(-1, se.shape[-1]))

		if self.args.n_levels_bias:
			pe_bias = pe[
				..., : self.args.n_levels_bias * self.args.n_features_per_level
			]
			assert zs[0].shape[0] == pe_bias.shape[0], "slice embedding should have the same batch size as pe_bias"
			results["log_bias"] = self.b_net(torch.cat(zs + [pe_bias], -1)).view(
				prefix_shape
			)

		if not self.args.no_pixel_variance:
			zs.append(z[..., 1:])
			results["log_var"] = self.sigma_net(torch.cat(zs, -1)).view(prefix_shape)
		
		check_nan(results)	
		return results
	
	
	@torch.autocast("cuda", cache_enabled=False)
	def forward(
		self,
		xyz: torch.Tensor,
		v: torch.Tensor,
		slice_idx: torch.Tensor,
	) -> Dict[str, Any]:
		# sample psf point
		batch_size = xyz.shape[0]
		n_samples = self.args.n_samples
		eps = 1e-5

		#NOTE: monte carlo sampling for PSF at each pixel location
		t = self.axisangle[slice_idx][:, None]
		xyz = xyz[:, None]
		if self.args.n_samples > 1:
			xyz_psf = torch.randn(
				batch_size, n_samples, 3, dtype=xyz.dtype, device=xyz.device
			) 
			psf_sigma = self.psf_sigma[slice_idx][:, None]
			xyz = xyz + xyz_psf * psf_sigma
			
		# transform points
		#NOTE: different from xyz_transformed in dataset if centering used
		xyz = ax_transform_points(
			t, xyz, self.trans_first
		).squeeze()

		# deform
		if self.args.deformable:
			xyz_ori = xyz
			de = self.deform_embedding(slice_idx)[:, None].expand(-1, n_samples, -1)
			xyz = self.deform_net(xyz, de)

		# inputs
		if self.args.n_features_slice:
			se = self.slice_embedding(slice_idx)[:, None].expand(-1, n_samples, -1).squeeze()
		else:
			se = None
		results = self.net_forward(xyz, se, values=v)
		if self.o_inr:
			# ground truth and slice embedding accumulated using trilinear interpolation
			v = results["values"].squeeze()
			# se = results['se']

		# output
		density = atleast_2d(results["density"])
		if "log_bias" in results:
			log_bias = atleast_2d(results["log_bias"])
			bias = log_bias.exp()
			bias_detach = bias.detach()
		else:
			log_bias = 0
			bias = 1
			bias_detach = 1
		if "log_var" in results:
			log_var = atleast_2d(results["log_var"])
			var = log_var.exp()
		else:
			log_var = 0
			var = 1
		# imaging
		if not self.args.no_slice_scale:
			c: Any = F.softmax(self.logit_coef, 0)[slice_idx] * self.n_slices
			if self.o_inr:
				_, c = self.inr.point2voxel(xyz, c, patchify=self.args.patchify)
		else:
			c = 1
		# unsqueeze feature dim if needed
		c = c.squeeze()

		v_out = (bias * density).mean(-1)
		v_out = c * v_out

		if not self.args.no_pixel_variance:
			var = (bias_detach * var).mean(-1)
			var = c.detach() * var
			var = var**2
			var += eps

		if not self.args.no_slice_variance and not self.o_inr :
			var = var + self.log_var_slice.exp()[slice_idx]
		if self.args.mse_only:
			losses = {D_LOSS: F.mse_loss(v_out, v)}
		else:
			# losses
			# eq (12)
			losses = {D_LOSS: ((v_out - v) ** 2 / (2 * var + eps)).mean()}		
			if not (self.args.no_pixel_variance and self.args.no_slice_variance):
				losses[S_LOSS] = 0.5 * var.log().mean()
				losses[DS_LOSS] = losses[D_LOSS] + losses[S_LOSS]
			if not self.args.no_transformation_optimization:
				losses[T_REG] = self.trans_loss(trans_first=self.trans_first)
			if self.args.n_levels_bias:
				losses[B_REG] = log_bias.mean() ** 2
			if self.args.deformable:
				losses[D_REG] = self.deform_reg(
					xyz, xyz_ori, de
				)  # deform_reg_autodiff(self.deform_net, xyz_ori, de)
			# image regularization
			losses[I_REG] = self.img_reg(density, xyz)
			
			# debug loss
			check_nan(losses)
				
		if self.args.patchify:
			self.inr.reset_patch()
		if self.args.profiling:
			print("max memory allocated: ", torch.cuda.max_memory_allocated() / 1024 ** 3, "GB")
			torch.cuda.reset_max_memory_allocated()
			
		return losses


	def trans_loss(self, trans_first: bool = True) -> torch.Tensor:
		x = RigidTransform(self.axisangle, trans_first=trans_first)
		y = RigidTransform(self.axisangle_init, trans_first=trans_first)
		err = y.inv().compose(x).axisangle(trans_first=trans_first)
		loss_R = torch.mean(err[:, :3] ** 2)
		loss_T = torch.mean(err[:, 3:] ** 2)
		return loss_R + 1e-3 * self.spatial_scaling * self.spatial_scaling * loss_T

	def img_reg(self, density, xyz):
		if self.args.image_regularization == "none":
			return torch.zeros((1,), dtype=density.dtype, device=density.device)

		if self.args.img_reg_autodiff:
			n_sample = 4
			xyz = xyz[:, :n_sample].flatten(0, 1).detach()
			xyz.requires_grad_()
			density, _, _ = self.inr(xyz)
			grad = (
				torch.autograd.grad((density.sum(),), (xyz,), create_graph=True)[0]
				/ self.spatial_scaling
			)
			grad2 = grad.pow(2)
		else:
			# eq (15). Won't work for O-INR since aren't extra PSF sampled points
			xyz = xyz * self.spatial_scaling
			d_density = density - torch.flip(density, (1,))
			dx2 = ((xyz - torch.flip(xyz, (1,))) ** 2).sum(-1) + 1e-6
			grad2 = d_density**2 / dx2

		if self.args.image_regularization == "TV":
			return grad2.sqrt().mean()
		elif self.args.image_regularization == "edge":
			return self.delta * (
				(1 + grad2 / (self.delta * self.delta)).sqrt().mean() - 1
			)
		elif self.args.image_regularization == "L2":
			return grad2.mean()
		else:
			raise ValueError("unknown image regularization!")

	def deform_reg(self, out, xyz, e):
		if True:  # use autodiff
			n_sample = 4
			x = xyz[:, :n_sample].flatten(0, 1).detach()
			e = e[:, :n_sample].flatten(0, 1).detach()

			x.requires_grad_()
			outputs = self.deform_net(x, e)
			grads = []
			out_sum = []
			for i in range(3):
				out_sum.append(outputs[:, i].sum())
				grads.append(
					torch.autograd.grad((out_sum[-1],), (x,), create_graph=True)[0]
				)
			jacobian = torch.stack(grads, -1)
			jtj = torch.matmul(jacobian, jacobian.transpose(-1, -2))
			I = torch.eye(3, dtype=jacobian.dtype, device=jacobian.device).unsqueeze(0)
			sq_residual = ((jtj - I) ** 2).sum((-2, -1))
			return torch.nan_to_num(sq_residual, 0.0, 0.0, 0.0).mean()
		else:
			out = out - xyz
			d_out2 = ((out - torch.flip(out, (1,))) ** 2).sum(-1) + 1e-6
			dx2 = ((xyz - torch.flip(xyz, (1,))) ** 2).sum(-1) + 1e-6
			dd_dx = d_out2.sqrt() / dx2.sqrt()
			return F.smooth_l1_loss(dd_dx, torch.zeros_like(dd_dx).detach(), beta=1e-3)
		

