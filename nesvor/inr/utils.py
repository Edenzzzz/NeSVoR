import torch
def atleast_2d(x: torch.Tensor):
	"""
	Ensure feature dim exists. Mostly to account for the removed PSF dim in 3d conv.
	"""
	if x.dim() == 1:
		x = x.unsqueeze(1)

	return x

def byte2mb(byte: int):
	return byte / 1024 / 1024

def unique(x, dim=None, return_index=True, return_inverse=False, return_counts=False):
	"""Unique elements of x and indices of those unique elements
	https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

	e.g.

	unique(tensor([
		[1, 2, 3],
		[1, 2, 4],
		[1, 2, 3],
		[1, 2, 5]
	]), dim=0)
	=> (tensor([[1, 2, 3],
				[1, 2, 4],
				[1, 2, 5]]),
		tensor([0, 1, 3]))
	"""
	unique, inverse, counts = torch.unique(
		x, sorted=True, return_inverse=True, dim=dim, return_counts=True)

	perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
						device=inverse.device)
	inv_flipped, perm = inverse.flip([0]), perm.flip([0])
	return_list = [unique]
	if return_index:
		return_list.append(inv_flipped.new_empty(unique.size(0)).scatter(0, inv_flipped, perm))
	if return_counts:
		return_list.append(counts)
	if return_inverse:
		return_list.append(inverse)
	return return_list

def freeze_params(params: iter):
	for param in params:
		param.requires_grad = False


def free_memory():
	import gc
	import torch
	gc.collect()
	torch.cuda.empty_cache()