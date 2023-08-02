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
import os
from .utils import *
USE_TORCH = True
	
if not USE_TORCH:
	try:
		import tinycudann as tcnn
	except:
		logging.warning("Fail to load tinycudann. Will use pytorch implementation.")
		USE_TORCH = True
mem_log = []
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
		dataset: Optional[PointDataset] = None,
	) -> None:
		"""
		Takes in discrete 3D coordinates within the bounding box 
		"""
		super().__init__()
		if TYPE_CHECKING:
			self.bounding_box: torch.Tensor
		self.register_buffer("bounding_box", bounding_box)
		self.dataset = dataset
		self.args = args

		# hash grid encoding
		base_resolution, n_levels = compute_resolution_nlevel(
			self.bounding_box,
			args.coarsest_resolution,
			args.finest_resolution,
			args.level_scale,
			spatial_scaling,
		)

		self.encoding = build_encoding(
			n_input_dims=3,
			otype="HashGrid",
			n_levels=n_levels,
			n_features_per_level=args.n_features_per_level,
			log2_hashmap_size=args.log2_hashmap_size,
			base_resolution=base_resolution,   
			per_level_scale=args.level_scale,   # @wenxuan: the resolution/grid size multiplier between levels, by which coords are multiplied 
												# Low res -> high res, low freq -> high freq
			dtype=args.dtype,
		)
		
		self.o_inr = args.o_inr
		# density net
		if self.o_inr:   
			self.density_net = volumeNet(
				None,
				None,
				inchannel=n_levels * args.n_features_per_level,
				outchannel=1 + args.n_features_z,
				learnable_wave=None,
				transform=None,
				mode=None
			)
			self.forward = self._volume_forward
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


	def points2volume(self, xyz, values=None):
		self.avg_neighbors = values != None
		if self.avg_neighbors:
			vol_in, xyz_idx_unique, xyz, v_gt = self.dataset.match_grid(xyz, values)
			self.v_gt = v_gt
			self.xyz_idx_unique = xyz_idx_unique
		else:
			# do not accumulate values for overlapping points
			vol_in, xyz_idx_unique, xyz, inv_idx = self.dataset.match_grid(xyz)
			self.xyz_idx_unique = xyz_idx_unique # original array -> unique value array. 
			self.duplicate_select = inv_idx # unique value array -> original array. shape = (npoints, )
		del values

		# hash grid encoding
		prefix_shape = vol_in.shape[:-1]
		vol_in = vol_in.reshape(-1, 3)
		vol_in = self.encoding(vol_in).reshape(prefix_shape + (-1,))
		pe = vol_in[xyz[:, 0], xyz[:, 1], xyz[:, 2]]

		if self.avg_neighbors:
			return pe, vol_in, xyz
		return pe, vol_in, xyz
	

	def _volume_forward(self,
						xyz: torch.Tensor,
						values: Optional[torch.Tensor] = None,
						):
		"""
		Args: 
			xyz: (npoints, 3)
			se: (npoints, 16)
			values: (npoints, ) GT density values NOTE: not used to accumulate gt currently
		"""
		if self.training and values is None:
			Warning.warn(" In training but values not provided to forward, overlapping intensities won't be averaged!")
			
		pe, vol_in, xyz_grid_unique = self.points2volume(xyz, values)

		# 3D conv
		if not self.training:
			pe = pe.to(dtype=vol_in.dtype)
		vol_in = vol_in.unsqueeze(0).permute(0, 4, 1, 2, 3) # (1, encoding_dim, x, y, z)
		z = self.density_net(vol_in); del vol_in
		z = z.permute(0, 2, 3, 4, 1).squeeze()
		
		# volume to points
		z = z[xyz_grid_unique[:, 0], xyz_grid_unique[:, 1], xyz_grid_unique[:, 2]]
		density = F.softplus(z[:, 0])
		assert len(density) == len(xyz_grid_unique), "each unique point should have a density value"
		
		
		# unique array -> original array; duplicates filled with the value of the first occurence
		if not self.avg_neighbors:
			pe = pe[self.duplicate_select]
			z = z[self.duplicate_select]
			density = density[self.duplicate_select]
			
		if self.training:
			return density, pe, z
		return density


	def forward(self, x: torch.Tensor, **kwargs):
		if not self.args.use_voxel:
			# normalize coordinates to [0, 1]
			x = (x - self.bounding_box[0]) / (self.bounding_box[1] - self.bounding_box[0])
			prefix_shape = x.shape[:-1] # (batch_size, n_psf_samples, 3)
			x = x.view(-1, x.shape[-1])
			pe = self.encoding(x)
		else:
			pe, _, _ = self.points2volume(x)
			# fill overlapping points with the value of the first occurence
			pe = pe[self.xyz_idx_unique][self.duplicate_select]


		# The density net takes encodings at all levels, 
		# but the other two MLPs split them
		if not self.training:
			pe = pe.to(dtype=x.dtype)
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
		if n_samples > 1:
			if isinstance(psf_sigma, torch.Tensor):
				psf_sigma = psf_sigma.view(-1, 1, 3)
			xyz_psf = torch.randn(
				xyz.shape[0], n_samples, 3, dtype=xyz.dtype, device=xyz.device
			)
			xyz = xyz[:, None] + xyz_psf * psf_sigma
		else:
			xyz = xyz[:, None]
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
		
		hash_path = os.path.join(self.args.log_dir, "hash_grid.pt")
		if args.save_hash:
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
		elif os.path.exists(hash_path):
			self.encoding = torch.load(hash_path)

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
		dataset: Optional[PointDataset] = None,
	) -> None:
		super().__init__()
		if "cpu" in str(args.device):  # CPU mode
			global USE_TORCH
			USE_TORCH = True
		else:
			# set default GPU for tinycudann
			torch.cuda.set_device(args.device)
		self.dataset = dataset
		self.spatial_scaling = spatial_scaling
		self.args = args
		self.n_slices = 0
		self.trans_first = True
		self.transformation = transformation
		self.psf_sigma = resolution2sigma(resolution, isotropic=False)
		self.delta = args.delta * v_mean
		self.build_network(bounding_box)
		self.to(args.device)
		
		if self.args.o_inr:
			self.args.n_samples = 1 # set PSF to perturbe only the current point
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

	def build_network(self, bounding_box) -> None:
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
		
		self.inr = INR(bounding_box, self.args, self.spatial_scaling, self.dataset)
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
		# map to voxel grid here 
		use_volume_net = isinstance(self.inr.density_net, volumeNet)

		density, pe, z = self.inr.forward(x, values=values)
		del x, values

		if use_volume_net:
			xyz_idx_unique = self.inr.xyz_idx_unique
		prefix_shape = density.shape
		results = {"density": density}

		zs = []
		if se is not None:
			if use_volume_net:
				se = se[xyz_idx_unique]
				if not self.inr.avg_neighbors:
					se = se[self.inr.duplicate_select]
			zs.append(se.reshape(-1, se.shape[-1]))

		if self.args.n_levels_bias:
			pe_bias = pe[
				..., : self.args.n_levels_bias * self.args.n_features_per_level
			]

			results["log_bias"] = self.b_net(torch.cat(zs + [pe_bias], -1)).view(
				prefix_shape
			)
			del pe_bias, pe

		if not self.args.no_pixel_variance:
			zs.append(z[..., 1:])
			results["log_var"] = self.sigma_net(torch.cat(zs, -1)).view(prefix_shape)

		return results
	
	@torch.autocast("cuda")
	def forward(
		self,
		xyz: torch.Tensor,
		v: torch.Tensor,
		slice_idx: torch.Tensor,
	) -> Dict[str, Any]:
		# sample psf point
		batch_size = xyz.shape[0]
		n_samples = self.args.n_samples
		#NOTE: monte carlo sampling for psf at each pixel location
		# see notes before eq. (7) in the paper
		xyz_psf = torch.randn(
			batch_size, n_samples, 3, dtype=xyz.dtype, device=xyz.device
		) 
		# psf = 1
		psf_sigma = self.psf_sigma[slice_idx][:, None]
		# transform points
		t = self.axisangle[slice_idx][:, None]

		#NOTE: different from xyz_transformed in dataset if centering used
		xyz = ax_transform_points(
			t, xyz[:, None] + xyz_psf * psf_sigma, self.trans_first
		).squeeze()

		# deform
		xyz_ori = xyz
		if self.args.deformable:
			de = self.deform_embedding(slice_idx)[:, None].expand(-1, n_samples, -1)
			xyz = self.deform_net(xyz, de)

		# inputs
		if self.args.n_features_slice:
			se = self.slice_embedding(slice_idx)[:, None].expand(-1, n_samples, -1)
		else:
			se = None
		results = self.net_forward(xyz, se, values=v)

		# select indices with non-duplicate coordinates
		if self.args.o_inr:
			xyz_idx_unique = self.inr.xyz_idx_unique
			se = se[xyz_idx_unique]
			slice_idx = slice_idx[xyz_idx_unique]
			if self.inr.avg_neighbors:
				v = self.inr.v_gt # averaged neighboring (overlapping) intensities
			else:
				# expand to original shape by over-selecting the first occurence of each unique coordinate
				v = v[xyz_idx_unique][self.inr.duplicate_select]
				se = se[self.inr.duplicate_select]
				density = density[self.inr.duplicate_select]
			
		# output
		density = results["density"]
		if "log_bias" in results:
			log_bias = results["log_bias"]
			bias = log_bias.exp()
			bias_detach = bias.detach()
		else:
			log_bias = 0
			bias = 1
			bias_detach = 1
		if "log_var" in results:
			log_var = results["log_var"]
			var = log_var.exp()
		else:
			log_var = 0
			var = 1
		# imaging
		if not self.args.no_slice_scale:
			c: Any = F.softmax(self.logit_coef, 0)[slice_idx] * self.n_slices
		else:
			c = 1

		v_out = (bias * density).mean(-1)
		v_out = c * v_out
		if not self.args.no_pixel_variance:
			# NOTE: The mean variance over PSF dim
			var = (bias_detach * var).mean(-1)
			var = c.detach() * var
			var = var**2
		if not self.args.no_slice_variance:
			var = var + self.log_var_slice.exp()[slice_idx]
		# losses
		# eq (12)
		losses = {D_LOSS: ((v_out - v) ** 2 / (2 * var)).mean()}
				
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

		
		global mem_log
		mem_log += [byte2mb(torch.cuda.memory_allocated())]
		if len(mem_log) == 10:
			with open(self.args.mem_log, "a") as f:
				f.write(f"CUDA memory cost:  {sum(mem_log) / len(mem_log)}, MB\n")
				mem_log.clear()
			
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
