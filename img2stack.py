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

train_df = pd.read_csv("kaggle_data/train.csv")

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


def show_slices(volume, title="Slices"):
    """ Function to display row of image slices """
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()
    h, w, l = volume.shape
    slice_0 = volume[h // 2, :, :]
    slice_1 = volume[:, w // 2, :]
    slice_2 = volume[:, :, l // 2]
    slices = [slice_0, slice_1, slice_2]
    
    fig, axes = plt.subplots(1, len(slices))
    fig.suptitle(title)
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
        
    return fig

def imgs2stack(reconstruct=True, out_dir="volumes", case=2, day=1):
    '''
    Put all (kaggle) images into one stack
    '''
    path_to_slices = os.path.join("kaggle_data", "train", f"case{case}", f"case{case}_day{day}", "scans")
    slice_files = sorted(os.listdir(path_to_slices))
    # Load the first slice to determine the shape
    first_slice = open_gray16(os.path.join(path_to_slices, slice_files[0]))
    slice_shape = first_slice.shape

    # Create an empty 3D volume array
    stacked = np.zeros((len(slice_files),) + slice_shape, dtype=np.float32)

    # Iterate over the slice files and populate the volume array
    for i, slice_file in enumerate(slice_files):
        slice_path = os.path.join(path_to_slices, slice_file)
        slice_img = open_gray16(slice_path)
        stacked[i, :, :] = slice_img

    # Create a NIfTI image object for the 3D volume
    volume_image = nib.Nifti1Image(stacked, affine=None) #?

    # Save the 3D volume as a compressed (.nii.gz) file
    out_path  = os.path.join(out_dir, f'case{case}_day{day}_stack.nii.gz')
    nib.save(volume_image, out_path)

    if reconstruct:
        # Run the reconstruction
        result = subprocess.run(["nesvor", "reconstruct",
                            "--input-stacks", out_path,
                            "--output-volume", os.path.join(out_dir, f"case{case}_day{day}_volume.nii.gz"),
                            "--bias-field-correction",
                            "--registration", "svort",
                            "--otsu-thresholding"
                            ])

def volume2stacks(paths, reconstruct=True, out_dir=None):
    """
    Extract 3 orthogonal stacks from ADNI volumes
    """
    slice_acq = TestSliceAcq(verbose=True)
    for path in paths:
        volume = nib.load(path)
        affine = volume.affine
        volume = volume.get_fdata()
        h, w, l = volume.shape

        # Extract 3 orthogonal stacks
        angles = [
            [1.57079633, 0., 0.],
            [0., 1.57079633, 0.],
            [0., 0., 1.57079633]
            ]
        stacks, transforms, volume, params = slice_acq.get_cg_recon_test_data(angles, volume)
        #save stacks
        out_dir = os.path.split(path)[0]
        name = os.path.split(path)[-1]
        stack_path = []
        for idx, (stack, transform) in enumerate(zip(stacks, transforms)):
            # affine = transform.matrix().squeeze()
            # affine = torch.concat([affine, torch.tensor([[0, 0, 0, 1]]).to(affine)])
            
            #switch to nii and save stack
            # stack = tensor2nii(stack, affine=affine.cpu().numpy())
            
            stack = tensor2nii(stack, affine=affine)
            path = os.path.join(out_dir, f"{name}_stack{idx}.nii.gz")
            nib.save(stack, path)
            stack_path.append(path)

        if reconstruct:
            rec_volume_path = os.path.join(out_dir, f"reconstruct_{name}_volume_h.nii.gz")
            result = subprocess.run(["nesvor", "reconstruct",  
                            "--input-stacks", *stack_path,
                            "--output-volume", rec_volume_path,
                            "--bias-field-correction",
                            "--registration", "svort",
                            # "--otsu-thresholding" #mask out everything
                            ])
        
        os.makedirs(os.path.join(out_dir, "quality_assess"), exist_ok=True)
        subprocess.run(["nesvor", "assess",
                        "--input-stacks", *stack_path,
                        "--output-json", os.path.join(out_dir, "quality_assess", f"{name}_assess.json")
                        ])
        reconstructed = nib.load(os.path.join(out_dir, f"reconstruct_{name}_volume_h.nii.gz")).get_fdata()
        fig = show_slices(reconstructed)
        fig.savefig(os.path.join(out_dir, f"reconstruct_{name}.png"))

if __name__ == "__main__":
    # imgs2stack()
    prefix = "MedicalImaging/adcp_alldata_location2/ADCP_harmonization"
    paths = sorted(os.listdir(prefix))[:5]
    paths = [os.path.join(prefix, path, "T1w_acpc_dc_restore_1mm.nii.gz") for path in paths]
    volume2stacks(paths)