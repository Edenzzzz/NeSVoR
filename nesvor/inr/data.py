from typing import Dict, List
import torch
import torch.nn.functional as F
from nesvor.image import Slice
from ..utils import gaussian_blur
from ..transform import RigidTransform, transform_points
from ..image import Volume, Slice
from typing import Tuple
from pytorch3d.structures import Volumes, Pointclouds
from pytorch3d.ops import add_pointclouds_to_volumes
import logging
import math
import numpy as np
from .utils import *


class PointDataset(object):
	def __init__(self, slices: List[Slice], args, num_patches=5) -> None:
		self.mask_threshold = 1  # args.mask_threshold
		self.args = args
		self.num_patches = num_patches
		self.patch_idx = np.random.randint(0, num_patches)

		xyz_all = []
		v_all = []
		slice_idx_all = []
		transformation_all = []
		resolution_all = []
		
		for i, slice in enumerate(slices):
			# slice.data is the slice: (1, h, w)
			# slice._mask: (1, h, w)
			# slice.xyz_masked_untransformed: (slice._mask.sum(), 3)
			# slice.v_masked: (slice._mask.sum(), )
			xyz = slice.xyz_masked_untransformed
			v = slice.v_masked                      # image data TODO: how is v preprocessed?
			slice_idx = torch.full(v.shape, i, device=v.device)
			xyz_all.append(xyz)
			v_all.append(v)
			slice_idx_all.append(slice_idx)
			transformation_all.append(slice.transformation)
			resolution_all.append(slice.resolution_xyz)

		self.xyz = torch.cat(xyz_all)               # (?, 3) NOTE: each slice has a different number of unmasked points
		self.v = torch.cat(v_all)                   # (?, )
		self.slice_idx = torch.cat(slice_idx_all)   # (?, )
		self.transformation = RigidTransform.cat(transformation_all) # (nslices, 3, 4)
		self.resolution = torch.stack(resolution_all, 0)
		self.count = self.v.shape[0]
		self.epoch = 0
		orig_sidelen = slices[0].shape_xyz.max()
		self.orig_vol_shape = torch.tensor([orig_sidelen, orig_sidelen, orig_sidelen]).to(self.resolution.device)
	
	def reset(self):
		self.patch_idx = np.random.randint(0, self.num_patches)

	# boundary of the voxel grid used during training.
	# Later when generating the volume, will use 10 times
	# the max resolution instead of 2
	@property
	def bounding_box(self) -> torch.Tensor:
		max_r = self.resolution.max()
		xyz_transformed = self.xyz_transformed
		xyz_min = xyz_transformed.amin(0) - 2 * max_r
		xyz_max = xyz_transformed.amax(0) + 2 * max_r
		bounding_box = torch.stack([xyz_min, xyz_max], 0)
		return bounding_box
	
	@property
	def inner_bbox(self) -> torch.Tensor:
		xyz_transformed = self.xyz_transformed
		bounding_box = torch.stack([xyz_transformed.amin(0), xyz_transformed.amax(0)], 0)
		return bounding_box
	

	@property
	def mean(self) -> float:
		q1, q2 = torch.quantile(
			self.v if self.v.numel() < 256 * 256 * 256 else self.v[: 256 * 256 * 256],
			torch.tensor([0.1, 0.9], dtype=self.v.dtype, device=self.v.device),
		)
		return self.v[torch.logical_and(self.v > q1, self.v < q2)].mean().item()

	def get_batch(self, batch_size: int, device, use_voxel_grid=False) -> Dict[str, torch.Tensor]:

		if self.count + batch_size > self.xyz.shape[0]:  # new epoch, shuffle data
			self.count = 0
			self.epoch += 1
			idx = torch.randperm(self.xyz.shape[0], device=device)
			# slice coordinates
			self.xyz = self.xyz[idx]

			self.v = self.v[idx]
			self.slice_idx = self.slice_idx[idx]
		# fetch a batch of data
		batch = {
			"xyz": self.xyz[self.count : self.count + batch_size],
			"v": self.v[self.count : self.count + batch_size],
			"slice_idx": self.slice_idx[self.count : self.count + batch_size],
		}
		self.count += batch_size
		return batch
	
	# multiply with the transformation matrix
	@property
	def xyz_transformed(self) -> torch.Tensor:
		return transform_points(self.transformation[self.slice_idx], self.xyz)

	#TODO: This doesn't seem to be the final output size?
	@property
	def mask(self) -> Volume:
		with torch.no_grad():
			resolution_min = self.resolution.min()
			resolution_max = self.resolution.max()
			xyz = self.xyz_transformed
			xyz_min = xyz.amin(0) - resolution_max * 10
			xyz_max = xyz.amax(0) + resolution_max * 10
			shape_xyz = ((xyz_max - xyz_min) / resolution_min).ceil().long()
			shape = (int(shape_xyz[2]), int(shape_xyz[1]), int(shape_xyz[0]))
			kji = ((xyz - xyz_min) / resolution_min).round().long()

			mask = torch.bincount(
				kji[..., 0]
				+ shape[2] * kji[..., 1]
				+ shape[2] * shape[1] * kji[..., 2],
				minlength=shape[0] * shape[1] * shape[2],
			)
			mask = mask.view((1, 1) + shape).float()
			mask_threshold = (
				self.mask_threshold
				* resolution_min**3
				/ self.resolution.log().mean().exp() ** 3
			)
			mask_threshold *= mask.sum() / (mask > 0).sum()
			assert len(mask.shape) == 5
			mask = (
				gaussian_blur(mask, (resolution_max / resolution_min).item(), 3)
				> mask_threshold
			)[0, 0]

			xyz_c = xyz_min + (shape_xyz - 1) / 2 * resolution_min
			return Volume(
				mask.float(),
				mask,
				RigidTransform(torch.cat([0 * xyz_c, xyz_c])[None], True),
				resolution_min,
				resolution_min,
				resolution_min,
			)

