
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

def tensor2nii(tensor, affine=None, save=True, name="phantom"):
    """ Helper to convert a tensor to a NIfTI image object """
    volume = nib.Nifti1Image(tensor.detach().cpu().numpy(), affine=affine)
    if save:
        nib.save(volume, f"{name}.nii.gz")
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
    

    def get_cg_recon_test_data(self, angles, volume=None, return_stack_as_list=True):
        vs = 240 #no error with vs = 64 
        gap = s_thick = 3#not the thickness in MRI scan
        res = 1
        res_s = 1.5
        n_slice = int((np.sqrt(3) * vs) / gap) + 4
        ss = int((np.sqrt(3) * vs) / res_s) + 4
        # print(f"Creating a volume of {(vs, vs, vs)} with resolution {res}")
        if volume is None:
            if self.phantom == "default":
                volume = phantom3d(n=vs, phantom="shepp_logan")
            else:
                volume = shepp_logan((vs, vs, vs))
            
        volume = torch.tensor(volume, dtype=torch.float32).cuda()[None, None].contiguous()
        psf = get_PSF(res_ratio=(res_s / res, res_s / res, s_thick / res)).cuda()


        stacks = []
        transforms = []
       
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
            tx = ty = torch.ones_like(tz) * 0.5 #pick middle coordinates? but why 0.5 not 0.5 * dim
            t = torch.stack((tx, ty, tz), -1)
            transform = RigidTransform(torch.cat((angle, t), -1), trans_first=True)
            # sample slices
            mat = mat_update_resolution(transform.matrix(), 1, res)
            slices = slice_acquisition(
                mat, volume, None, None, psf, (ss, ss), res_s / res, False, False
            )
            if return_stack_as_list:
                slices = slices.squeeze()
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


    def test_cg_recon(self, pick_best_angles=False, volume=None, angles=None) -> list or torch.Tensor:
        #enumerate all possible angle triplets and get the top 5 with lowest error
        if pick_best_angles:
            self.angle_triplets = itertools.combinations(self.all_angles, 3)
            self.angle_triplets = np.array(list(self.angle_triplets))
            print(f"Testing a total of {len(self.angle_triplets)} angle triplets! LOL")
            for angles in self.angle_triplets:
                self.test_cg_recon_single(angles, volume)
            
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
            volume_ = self.test_cg_recon_single(angles, volume=volume) 
            return volume_  
        # self.assert_tensor_close(volume_, volume, atol=3e-5, rtol=1e-5)
    

    def test_cg_recon_single(self, angles, volume=None):
        #get input
        slices, transforms, volume, params = self.get_cg_recon_test_data(angles, volume, return_stack_as_list=False) #svort requires concated slices
        srr = SRR(n_iter=20, tol=1e-8)
        theta = mat_update_resolution(transforms.matrix(), 1, params["res_r"])
        volume_ = srr(theta, slices, volume, params)
        
        #log error metrics
        self.log_metrics(volume_, volume, angles)
        return volume_
    
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
            print(f"Mismatches: {self.mismatches[-1]}/{self.total}({self.mismatches[-1] / self.total * 100}%), \
                    max error: {self.max_err[-1]}, mean error: {self.mean_err[-1]}"
                )
            print("----------------------------------------------------------")

        