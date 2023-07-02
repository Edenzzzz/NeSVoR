import numpy as np
import cv2 
import os
import nibabel as nib
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tests.slice_acquisition.test_slice_acq import TestSliceAcq
import torch
from nesvor.image import Stack
from typing import List, Tuple, Union

def open_gray16(path, normalize=True, to_rgb=False):
    """ Helper to open files """
    if normalize:
        if to_rgb:
            return np.tile(np.expand_dims(cv2.imread(path, cv2.IMREAD_ANYDEPTH)/65535., axis=-1), 3)
        else:
            return cv2.imread(path, cv2.IMREAD_ANYDEPTH)/65535.
    else:
        if to_rgb:
            return np.tile(np.expand_dims(cv2.imread(path, cv2.IMREAD_ANYDEPTH), axis=-1), 3)
        else:
            return cv2.imread(path, cv2.IMREAD_ANYDEPTH)


def tensor2nii(tensor, affine=None):
    """ Helper to convert a tensor to a NIfTI image object """
    return nib.Nifti1Image(tensor.detach().cpu().numpy(), affine=affine)


def show_slices(volume, volume_=None, compare=False):
    """ Function to display row of image slices """
    if isinstance(volume, torch.Tensor):
        volume = volume.squeeze().detach().cpu().numpy()
    if isinstance(volume_, torch.Tensor):
        volume_ = volume_.squeeze().detach().cpu().numpy()

    if compare:
        assert volume_ is not None, "Please provide a volume to compare to"
        fig, axes = plt.subplots(2, 3)
        plot_slices_helper(volume, "Original", fig, axes[0])
        plot_slices_helper(volume_, "Reconstructed", fig, axes[1])
    else:
        fig = plot_slices_helper(volume, "Reconstructed")

    return fig


def plot_slices_helper(volume, title="", fig=None, axes=None):
    h, w, l = volume.shape
    
    slice_0 = volume[h // 2, :, :]
    slice_1 = volume[:, w // 2, :]
    slice_2 = volume[:, :, l // 2]
    slices = [slice_0, slice_1, slice_2]

    if fig is None:
        fig, axes = plt.subplots(1, len(slices))
    axes[0].set_title(title, loc="center")

    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
    
    return fig 

def volume2stacks(paths, out_dir=None, gap=3, stack_res=0.8, source_res=0.5469):
    """
    Extract 3 orthogonal stacks from volumes and run reconstruction
    """
    slice_acq = TestSliceAcq(verbose=True)
    for path in paths:
        volume = nib.load(path)
        affine = volume.affine
        volume = torch.tensor(volume.get_fdata())
        h, w, l = volume.shape

        # Extract 3 orthogonal stacks
        angles = [
            [1.57079633, 0., 0.], #Got 90 degree LOL after all the exhaustive search for angles
            [0., 1.57079633, 0.],
            [0., 0., 1.57079633]
            ]
        stacks, transforms, volume, params = \
            slice_acq.get_cg_recon_test_data(angles, 
                                        volume,
                                        stack_res=stack_res, 
                                        source_res=1.5, #TODO: This doesn't appear to be the source resolution, or why must it be 1.5
                                                        #to produce good results?
                                        return_stack_as_list=True
                                        )
        
        #save stacks
        out_dir = "stacks"
        name = "feta" if any(["feta" in path for path in paths]) else "adni"
        subject_id = path.split("/")[-1][:7]
        stack_path = []
        for idx, (stack, transform) in enumerate(zip(stacks, transforms)):
            # stack = tensor2nii(stack, affine=affine)
            # path = os.path.join(out_dir, f"{name}_stack{idx}.nii.gz")
            # nib.save(stack, path)
            # stack_path.append(path)

            stack = Stack(stack, transformation=transform, resolution_x=stack_res, 
                          resolution_y=stack_res,
                          gap=gap,
                          thickness=gap,
                          ).get_volume()
            
            path = os.path.join(out_dir, f"{name}_{subject_id}_stack{idx}_gap={gap}.nii.gz")
            stack.save(path) #will squeeze the volume here
            stack_path += [path]
        
        
        out_dir = "volumes"
        rec_volume_path = os.path.join(out_dir, f"{name}_{subject_id}_rec_volume.nii.gz")
        result = subprocess.run(["nesvor", "reconstruct",  
                        "--input-stacks", *stack_path,
                        "--output-volume", rec_volume_path,
                        "--thickness", str(gap), str(gap), str(gap),
                        "--bias-field-correction",
                        "--registration", "svort",
                        "--metric", "ncc",  #assess stack quality
                        "--output-resolution", str(0.8) #default=0.8
                        ])
        

        reconstructed = nib.load(os.path.join(out_dir, f"{name}_{subject_id}_rec_volume.nii.gz")).get_fdata()
        fig = show_slices(volume, reconstructed)
        fig.savefig(os.path.join(out_dir, f"{name}_{subject_id}_rec.png"))


if __name__ == "__main__":
    # imgs2stack()
    # prefix = "MedicalImaging/adcp_alldata_location2/ADCP_harmonization"
    # paths = sorted(os.listdir(prefix))[:5]
    # paths = [os.path.join(prefix, path, "T1w_acpc_dc_restore_1mm.nii.gz") for path in paths]
    paths = ["feta/sub-001/anat/sub-001_rec-mial_T2w.nii.gz"]
    volume2stacks(paths)