
from tests import TestCaseNeSVoR
from nesvor.transform import RigidTransform, mat_update_resolution
from nesvor.slice_acquisition import slice_acquisition
from nesvor.utils import get_PSF
from nesvor.svort.models import SRR
from tests.phantom3d import phantom3d
import torch
import numpy as np
import nibabel as nib
from sigpy import shepp_logan
import itertools
import os
from typing import List
from nesvor.image import Stack


def pad_to_length(volume, length):
    """ Helper to pad a volume to a given length """
    #volume could be unsqueezed
    padded = torch.zeros(length, length, length).to(volume.device)
    padded[: volume.shape[0], : volume.shape[1], : volume.shape[2]] = volume
    if padded.shape != volume.shape:
        print(f"Padded {volume.shape} to {padded.shape}")
        
    return padded


def tensor2nii(tensor, affine=None, save=True, name="phantom", path=""):
    """ Helper to convert a tensor to a NIfTI image object """
    if len(tensor.shape) > 3:
        tensor = tensor.squeeze()
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    if isinstance(affine, torch.Tensor):
        affine = affine.detach().cpu().numpy()

    volume = nib.Nifti1Image(tensor, affine=affine)
    if save:
        nib.save(volume, os.path.join(path, f"{name}.nii.gz"))
    return volume


class TestSliceAcq(TestCaseNeSVoR):
    def __init__(self, phantom="default", verbose=False) -> None:
        self.phantom = phantom
        #specified by the authors
        self.all_angles = [
            [0, 0, 0],
            [np.pi / 2, 0, 0],
            [0, np.pi / 2, 0],
            [0, 0, np.pi / 2],
            [np.pi / 4, np.pi / 4, 0],
            [0, np.pi / 4, np.pi / 4],
            [np.pi / 4, 0, np.pi / 4],
            [np.pi / 3, np.pi / 3, 0],
            [0, np.pi / 3, np.pi / 3],
            [np.pi / 3, 0, np.pi / 3],
            [2 * np.pi / 3, 2 * np.pi / 3, 0],
            [0, 2 * np.pi / 3, 2 * np.pi / 3],
            [2 * np.pi / 3, 0, 2 * np.pi / 3],
            [np.pi / 5, np.pi / 5, 0],
            [0, np.pi / 5, np.pi / 5],
            [np.pi / 5, 0, np.pi / 5],
        ]
        self.reconstructed = []
        self.max_err = []
        self.mean_err = []
        self.mismatches = []
        self.volume = None
        self.total = None
        self.tested_angles = []
        self.verbose = verbose

        #save both volumes for comparison
        if not os.path.exists("phantom.nii.gz") or not os.path.exists("sigpy_phantom.nii.gz"):
            TestSliceAcq.compare_phantoms()
    
    
    @staticmethod     
    def compare_phantoms():
        vs = 240
        original = phantom3d(n=vs, phantom="shepp_logan")
        mine = shepp_logan((vs, vs, vs))
        tensor2nii(torch.tensor(original, dtype=torch.float32), name="phantom")
        tensor2nii(torch.tensor(mine, dtype=torch.float32), name="sigpy_phantom")
    

    def get_cg_recon_test_data(self, angles,
                                volume: torch.Tensor=None,
                                gap=3,
                                stack_res=1,
                                simulated_res=1.5,
                                return_stack_as_list=True,
                                ):
        """
        Args:
            angles: list of angles to use for slice acquisition
            gap: slice thickness(gap) in the stacks
            stack_res: in-plane resolution of the stacks along x and y axes
            simulated_res: in-plane resolution of the simulated (super-resolved) volume. 
            volume: if None, a phantom will be created.
        """

        vs = 240 if volume is None else max(volume.shape) # z axis

        s_thick = gap
        res = stack_res
        res_s = simulated_res 
        n_slice = int((np.sqrt(3) * vs) / gap) + 4
        ss = int((np.sqrt(3) * vs) / res_s) + 4
        
        if volume is None:
            
            print(f"Creating a Phantom of {(vs, vs, vs)} with resolution {res}")
            if self.phantom == "default":
                volume = phantom3d(n=vs, phantom="shepp_logan")
            else:
                volume = shepp_logan((vs, vs, vs))
        else:
            print('Using provided volume')
            if stack_res == 1 and simulated_res == 1.5:
                Warning("Real volume is providing but still specifying default phantom resolution")


        volume = torch.tensor(volume, dtype=torch.float32).cuda().contiguous()
        volume = pad_to_length(volume, vs)
        volume = volume.unsqueeze(0).unsqueeze(0) #preprocess for svort
        
        psf = get_PSF(res_ratio=(res_s / res, res_s / res, s_thick / res)).cuda()

        stacks = []
        transforms = []
        
        #get slices for each triplet of angles
        for i in range(len(angles)):
            angle = (
                torch.tensor([angles[i]], dtype=torch.float32)
                .cuda()
                .expand(n_slice, -1)
            )
            tz = (
                torch.arange(0, n_slice, dtype=torch.float32).cuda()
                - (n_slice - 1) / 2.0
            ) * gap
            tx = ty = torch.ones_like(tz) * 0.5 #these are some tricky transformations after this
            t = torch.stack((tx, ty, tz), -1)
            transform = RigidTransform(torch.cat((angle, t), -1), trans_first=True)
            # sample slices
            mat = mat_update_resolution(transform.matrix(), 1, res)
            slices = slice_acquisition(
                mat, volume, None, None, psf, (ss, ss), res_s / res, False, False
            )
            stacks.append(slices) #shape: (h, 1, w, l)
            transforms.append(transform)

        params = {
            "psf": psf,
            "slice_shape": (ss, ss),
            "res_s": res_s,
            "res_r": res,
            "interp_psf": False,
            "volume_shape": (vs, vs, vs),
        }
        

        if return_stack_as_list:
            return stacks, transforms, volume, params
        
        return torch.cat(stacks), RigidTransform.cat(transforms), volume, params


    def test_svort_recon(self, pick_best_angles=False, volume=None, angles=None) -> list or torch.Tensor:
        
        #enumerate all possible angle triplets and get the top 5 with lowest error
        if pick_best_angles:
            self.angle_triplets = itertools.combinations(self.all_angles, 3)
            self.angle_triplets = np.array(list(self.angle_triplets))
            print(f"Testing a total of {len(self.angle_triplets)} angle triplets! No!!!")
            for angles in self.angle_triplets:
                self.test_svort_recon_single(angles, volume)
            
            #pick the angles with lowest error    
            sorted_indices = sorted(range(len(self.reconstructed)),
                                     key=lambda k: (self.mismatches[k], self.mean_err)
            )
            #only pick from the best 5 
            sorted_indices = sorted_indices[:5]
            print("First 5 best angle triplets:",
                   self.angle_triplets[sorted_indices]
                   )
            return sorted_indices
        
        else:
            if angles is None:
                angles = self.all_angles
            volume_ = self.test_svort_recon_single(angles, volume=volume) 
            return volume_  
        # self.assert_tensor_close(volume_, volume, atol=3e-5, rtol=1e-5)
    

    def test_svort_recon_single(self, angles, volume=None):
        #get input
        slices, transforms, volume, params = self.get_cg_recon_test_data(angles, volume, return_stack_as_list=False) #svort requires concated slices
        srr = SRR(n_iter=20, tol=1e-8)
        theta = mat_update_resolution(transforms.matrix(), 1, params["res_r"])
        
        #NOTE: check that each slice stack has the same rotation matrix
        # compare = (theta[1, :, :3] == theta[:, :, :3]).all(dim=-1).all(dim=-1) #shape: (n_slices *)
        # assert compare[:compare.shape[0] // 3].all()
        volume_ = srr(theta, slices, volume, params)
        
        #log error metrics
        self.log_metrics(volume_, volume, angles)
        return volume_
    

    def test_nesvor_recon(self, volume=None, angles=None, stack_res=0.8, source_res=0.5469, name="feta") -> list or torch.Tensor:
        stacks, transforms, volume, params = \
        stacks, transforms, volume, params = \
            self.get_cg_recon_test_data(angles, 
                                    volume,
                                    stack_res=stack_res, 
                                    simulated_res=source_res,
                                    return_stack_as_list=True
                                    )
        
        #save stacks
        out_dir = "stacks"
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
                          thickness=stack_res,
                          ).get_volume()
            
            path = os.path.join(out_dir, f"{name}_{subject_id}_stack{idx}_gap={gap}.nii.gz")
            stack.save(path) #will squeeze the volume here
            stack_path += [path]
        
        
        out_dir = "volumes"
        rec_volume_path = os.path.join(out_dir, f"{name}_{subject_id}_rec_volume.nii.gz")
        result = subprocess.run(["nesvor", "reconstruct",  
                        "--input-stacks", *stack_path,
                        "--output-volume", rec_volume_path,
                        "--bias-field-correction",
                        "--registration", "svort",
                        "--metric", "ncc",  #assess stack quality
                        "--output-resolution", str(source_res)
                        ])
        

        reconstructed = nib.load(os.path.join(out_dir, f"{name}_{subject_id}_rec_volume.nii.gz")).get_fdata()
        fig = show_slices(reconstructed)
        fig.savefig(os.path.join(out_dir, f"{name}_{subject_id}_rec.png"))

    
    #compute error metrics and append to tester
    def log_metrics(self, reconstructed, volume, angles):
        self.mismatches += [torch.sum(torch.abs(reconstructed - volume) > 3e-5 + 1e-5 * volume)]
        self.max_err += [torch.max(torch.abs(reconstructed - volume))]
        self.reconstructed += [reconstructed.squeeze()]
        self.mean_err += [torch.mean(torch.abs(reconstructed - volume))]
        self.tested_angles += [angles]
        if self.volume is None:
            self.volume = volume
            self.total = torch.prod(torch.tensor(volume.shape))

        if self.verbose:
            print("Testing angles:", angles)
            print(f"Mismatches: {self.mismatches[-1]}/{self.total} ({(self.mismatches[-1] / self.total * 100).round()}%), \
                    \nmax error: {self.max_err[-1]}\nmean error: {self.mean_err[-1]}"
                )
            print("----------------------------------------------------------")

    
