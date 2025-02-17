from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..svr.reconstruction import cg
from ..transform import (
    RigidTransform,
    mat_update_resolution,
    ax_update_resolution,
    mat2axisangle,
    point2mat,
    mat2point,
)
from .attention import TransformerEncoder, PositionalEncoding, ResNet
from ..slice_acquisition import slice_acquisition, slice_acquisition_adjoint
from ..utils import gaussian_blur

# main models
import nibabel as nib

def tensor2nii(tensor, affine=None, save=True, name="phantom"):
    """ Helper to convert a tensor to a NIfTI image object """
    if len(tensor.shape) > 3:
        tensor = tensor.squeeze()
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    if isinstance(affine, torch.Tensor):
        affine = affine.detach().cpu().numpy()

    volume = nib.Nifti1Image(tensor, affine=affine)
    if save:
        nib.save(volume, f"{name}.nii.gz")
    return volume


USE_MASK = True


class SVoRT(nn.Module):
    def __init__(self, n_iter=3, iqa=True, vol=True, pe=True, dropout=0.0):
        super().__init__()
        self.n_iter = n_iter
        self.vol = vol
        self.pe = pe
        self.iqa = iqa and vol
        self.attn = None
        self.iqa_score = None

        svrnet_list = []
        for i in range(self.n_iter):
            svrnet_list.append(
                SVRtransformer(
                    n_res=50,
                    n_layers=4,
                    n_head=4 * 2,
                    d_in=9 + 2,
                    d_out=9,
                    d_model=256 * 2,
                    d_inner=512 * 2,
                    dropout=dropout,
                    res_d_in=4 if (i > 0 and self.vol) else 3,
                )
            )
        self.svrnet = nn.ModuleList(svrnet_list)
        if iqa:
            self.srrnet = SRRtransformer(
                n_res=34,
                n_layers=4,
                n_head=4,
                d_in=8,
                d_out=1,
                d_model=256,
                d_inner=512,
                dropout=dropout,
            )

    def forward(self, data):
        params = {
            "psf": data["psf_rec"],
            "slice_shape": data["slice_shape"],
            "res_s": data["resolution_slice"],
            "res_r": data["resolution_recon"],
            "volume_shape": data["volume_shape"],
        }

        transforms = RigidTransform(data["transforms"])
        stacks = data["stacks"]
        positions = data["positions"]

        thetas = []
        volumes = []
        trans = []

        if not self.pe:
            transforms = RigidTransform(transforms.axisangle() * 0)
            positions = positions * 0  # + data["slice_thickness"]

        theta = mat2point(
            transforms.matrix(), stacks.shape[-1], stacks.shape[-2], params["res_s"]
        )
        volume = None
        mask_stacks = gaussian_blur(stacks, 1.0, 3) > 0 if USE_MASK else None

        for i in range(self.n_iter):
            theta, attn = self.svrnet[i](
                theta,
                stacks,
                positions,
                None if ((volume is None) or (not self.vol)) else volume.detach(),
                params,
            )

            thetas.append(theta)

            _trans = RigidTransform(point2mat(theta))
            trans.append(_trans)

            with torch.no_grad():
                mat = mat_update_resolution(
                    _trans.matrix().detach(), 1, params["res_r"]
                )
                volume = slice_acquisition_adjoint(
                    mat,
                    params["psf"],
                    stacks,
                    mask_stacks,
                    None,
                    params["volume_shape"],
                    params["res_s"] / params["res_r"],
                    False,
                    equalize=True,
                )
                ax = mat2axisangle(_trans.matrix())
                ax = ax_update_resolution(ax, 1, params["res_s"])
            if self.iqa:
                volume, iqa_score = self.srrnet(
                    ax, mat, stacks, volume, params, positions
                )
                self.iqa_score = iqa_score.detach()
            volumes.append(volume)

        self.attn = attn.detach()

        return trans, volumes, thetas


class SVoRTv2(nn.Module):
    def __init__(self, n_iter=4, iqa=True, vol=True, pe=True):
        super().__init__()
        self.vol = vol
        self.pe = pe
        self.iqa = iqa and vol
        self.attn = None
        self.iqa_score = None
        self.n_iter = n_iter

        self.svrnet1 = SVRtransformerV2(
            n_layers=4,
            n_head=4 * 2,
            d_in=9 + 2,
            d_out=9,
            d_model=256 * 2,
            d_inner=512 * 2,
            dropout=0.0,
            n_channels=1,
        )

        self.svrnet2 = SVRtransformerV2(
            n_layers=4 * 2,
            n_head=4 * 2,
            d_in=9 + 2,
            d_out=9,
            d_model=256 * 2,
            d_inner=512 * 2,
            dropout=0.0,
            n_channels=2,
        )

        if iqa:
            self.srr = SRR(n_iter=2)

    def forward(self, data):
        params = {
            "psf": data["psf_rec"],
            "slice_shape": data["slice_shape"],
            "res_s": data["resolution_slice"],
            "res_r": data["resolution_recon"],
            "volume_shape": data["volume_shape"],
        }

        transforms = RigidTransform(data["transforms"])
        stacks = data["stacks"]
        positions = data["positions"]

        thetas = []
        volumes = []
        trans = []

        if not self.pe:
            transforms = RigidTransform(transforms.axisangle() * 0)
            positions = positions * 0  # + data["slice_thickness"]

        theta = mat2point(
            transforms.matrix(), stacks.shape[-1], stacks.shape[-2], params["res_s"]
        )
        volume = None
        mask_stacks = gaussian_blur(stacks, 1.0, 3) > 0 if USE_MASK else None

        for i in range(self.n_iter):
            svrnet = self.svrnet2 if i else self.svrnet1
            theta, iqa_score, attn = svrnet(
                theta,
                stacks,
                positions,
                None if ((volume is None) or (not self.vol)) else volume.detach(),
                params,
            )
            thetas.append(theta)
            _trans = RigidTransform(point2mat(theta))
            trans.append(_trans)
            with torch.no_grad():
                mat = mat_update_resolution(
                    _trans.matrix().detach(), 1, params["res_r"]
                )
                volume = slice_acquisition_adjoint(
                    mat,
                    params["psf"],
                    stacks,
                    mask_stacks,
                    None,
                    params["volume_shape"],
                    params["res_s"] / params["res_r"],
                    False,
                    equalize=True,
                )
            if self.iqa:
                volume = self.srr(
                    mat,
                    stacks,
                    volume,
                    params,
                    iqa_score.view(-1, 1, 1, 1),
                    mask_stacks,
                    None,
                )
                self.iqa_score = iqa_score.detach()
            volumes.append(volume)
        self.attn = attn.detach()
        return trans, volumes, thetas


# transformers


class SRRtransformer(nn.Module):
    def __init__(
        self,
        n_res=34,
        n_layers=4,
        n_head=4,
        d_in=8,
        d_out=1,
        d_model=256,
        d_inner=512,
        dropout=0.1,
    ):
        super().__init__()
        self.srr = SRR(n_iter=2)
        self.img_encoder = ResNet(
            n_res=n_res, d_model=d_model, pretrained=False, d_in=2
        )
        self.pos_emb = PositionalEncoding(d_model, d_in)
        self.encoder = TransformerEncoder(
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_model // n_head,
            d_v=d_model // n_head,
            d_model=d_model,
            d_inner=d_inner,
            dropout=dropout,
        )
        self.fc = nn.Linear(d_model, d_out)

    def forward(self, theta, transforms, slices, volume, params, idx):
        mask_stacks = gaussian_blur(slices, 1.0, 3) > 0 if USE_MASK else None
        slices_est = slice_acquisition(
            transforms,
            volume,
            None,
            mask_stacks,
            params["psf"],
            params["slice_shape"],
            params["res_s"] / params["res_r"],
            False,
            False,
        )
        idx = torch.cat((theta, idx), -1)
        x = torch.cat((slices, slices_est), 1)
        pe = self.pos_emb(idx)
        x = self.img_encoder(x)
        x, _ = self.encoder(x, pe)
        x = self.fc(x)
        x = F.softmax(x, dim=0) * x.shape[0]
        x = torch.clamp(x, max=3.0)
        volume = self.srr(
            transforms, slices, volume, params, x.view(-1, 1, 1, 1), mask_stacks, None
        )
        return volume, x


class SVRtransformer(nn.Module):
    def __init__(
        self,
        n_res=34,
        n_layers=4,
        n_head=4,
        d_in=8,
        d_out=6,
        d_model=256,
        d_inner=512,
        dropout=0.1,
        res_d_in=3,
        res_scale=1,
    ):
        super().__init__()
        if isinstance(n_res, nn.Module):
            self.img_encoder = n_res
        else:
            self.img_encoder = ResNet(
                n_res=n_res, d_model=d_model, pretrained=False, d_in=res_d_in
            )
        self.pos_emb = PositionalEncoding(d_model, d_in)
        self.encoder = TransformerEncoder(
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_model // n_head,
            d_v=d_model // n_head,
            d_model=d_model,
            d_inner=d_inner,
            dropout=dropout,
        )
        self.fc = nn.Linear(d_model, d_out)
        self.res_scale = res_scale
        self.res_d_in = res_d_in

    def pos_augment(self, slices, slices_est):
        n, _, h, w = slices.shape
        y = torch.linspace(-(h - 1) / 256, (h - 1) / 256, h, device=slices.device)
        x = torch.linspace(-(w - 1) / 256, (w - 1) / 256, w, device=slices.device)
        y, x = torch.meshgrid(y, x, indexing="ij")  # hxw
        if slices_est is not None:
            slices = torch.cat(
                [
                    slices,
                    slices_est,
                    y.view(1, 1, h, w).expand(n, -1, -1, -1),
                    x.view(1, 1, h, w).expand(n, -1, -1, -1),
                ],
                1,
            )
        else:
            if self.res_d_in == 3:
                slices = torch.cat(
                    [
                        slices,
                        y.view(1, 1, h, w).expand(n, -1, -1, -1),
                        x.view(1, 1, h, w).expand(n, -1, -1, -1),
                    ],
                    1,
                )
            else:
                slices = torch.cat(
                    [
                        slices,
                        0 * slices,
                        y.view(1, 1, h, w).expand(n, -1, -1, -1),
                        x.view(1, 1, h, w).expand(n, -1, -1, -1),
                    ],
                    1,
                )
        return slices

    def forward(self, theta, slices, pos, volume, params):
        y = volume
        if y is not None:
            with torch.no_grad():
                transforms = mat_update_resolution(point2mat(theta), 1, params["res_r"])
                y = slice_acquisition(
                    transforms,
                    y,
                    None,
                    None,
                    params["psf"],
                    params["slice_shape"],
                    params["res_s"] / params["res_r"],
                    False,
                    False,
                )
        pos = torch.cat((theta, pos), -1)
        pe = self.pos_emb(pos)
        slices = self.pos_augment(slices, y)
        x = self.img_encoder(slices)
        x, attn = self.encoder(x, pe)
        x = self.fc(x)
        return theta + x * self.res_scale, attn


class SVRtransformerV2(nn.Module):
    def __init__(
        self,
        n_res=50,
        n_layers=4,
        n_head=4,
        d_in=8,
        d_out=6,
        d_model=256,
        d_inner=512,
        dropout=0.1,
        n_channels=2,
    ):
        super().__init__()
        # self.img_encoder = ViT(channels=n_channels, num_classes=d_model)
        self.img_encoder = ResNet(
            n_res=n_res, d_model=d_model, pretrained=False, d_in=n_channels + 2
        )
        self.pos_emb = PositionalEncoding(d_model, d_in)
        self.encoder = TransformerEncoder(
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_model // n_head,
            d_v=d_model // n_head,
            d_model=d_model,
            d_inner=d_inner,
            dropout=dropout,
            activation_attn="softmax",
            activation_ff="gelu",
            prenorm=False,
        )
        self.fc = nn.Linear(d_model, d_out)
        self.fc_score = nn.Linear(d_model, 1)

    def pos_augment(self, slices, slices_est):
        n, _, h, w = slices.shape
        y = torch.linspace(-(h - 1) / 256, (h - 1) / 256, h, device=slices.device)
        x = torch.linspace(-(w - 1) / 256, (w - 1) / 256, w, device=slices.device)
        y, x = torch.meshgrid(y, x, indexing="ij")  # hxw
        if slices_est is not None:
            slices = torch.cat(
                [
                    slices,
                    slices_est,
                    y.view(1, 1, h, w).expand(n, -1, -1, -1),
                    x.view(1, 1, h, w).expand(n, -1, -1, -1),
                ],
                1,
            )
        else:
            slices = torch.cat(
                [
                    slices,
                    y.view(1, 1, h, w).expand(n, -1, -1, -1),
                    x.view(1, 1, h, w).expand(n, -1, -1, -1),
                ],
                1,
            )
        return slices

    def forward(self, theta, slices, pos, volume, params, attn_mask=None):
        y = volume
        if y is not None:
            with torch.no_grad():
                transforms = mat_update_resolution(point2mat(theta), 1, params["res_r"])
                y = slice_acquisition(
                    transforms,
                    y,
                    None,
                    None,
                    params["psf"],
                    params["slice_shape"],
                    params["res_s"] / params["res_r"],
                    False,
                    False,
                )
        pos = torch.cat((theta, pos), -1)
        pe = self.pos_emb(pos)
        if isinstance(self.img_encoder, ResNet):
            slices = self.pos_augment(slices, y)
        else:  # ViT
            if y is not None:
                slices = torch.cat([slices, y], 1)
        x = self.img_encoder(slices)
        x, attn = self.encoder(x, pe, attn_mask)
        dtheta = self.fc(x)

        score = self.fc_score(x)
        score = F.softmax(score, dim=0) * score.shape[0]
        score = torch.clamp(score, max=3.0)

        return theta + dtheta, score, attn


class SRR(nn.Module):
    def __init__(
        self,
        n_iter: int = 10,
        tol: float = 0.0,
        output_relu: bool = True,
    ) -> None:
        super().__init__()
        self.n_iter = n_iter
        self.tol = tol
        self.output_relu = output_relu

    def forward(
        self,
        transforms: torch.Tensor,
        slices: torch.Tensor,
        volume: torch.Tensor,
        params: Dict,
        p: Optional[torch.Tensor] = None,
        slices_mask: Optional[torch.Tensor] = None,
        vol_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        At = lambda x: self.At(transforms, x, params, slices_mask, vol_mask)
        AtA = lambda x: self.AtA(transforms, x, p, params, slices_mask, vol_mask)

        b = At(slices * p if p is not None else slices)
        volume = cg(AtA, b, volume, self.n_iter, self.tol)

        if self.output_relu:
            return F.relu(volume, True)
        else:
            return volume

    def A(
        self,
        transforms: torch.Tensor,
        x: torch.Tensor,
        params: Dict,
        slices_mask: Optional[torch.Tensor] = None,
        vol_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return slice_acquisition(
            transforms,
            x,
            vol_mask,
            slices_mask,
            params["psf"],
            params["slice_shape"],
            params["res_s"] / params["res_r"],
            False,
            False,
        )

    def At(
        self,
        transforms: torch.Tensor,
        x: torch.Tensor,
        params: Dict,
        slices_mask: Optional[torch.Tensor] = None,
        vol_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return slice_acquisition_adjoint(
            transforms,
            params["psf"],
            x,
            slices_mask,
            vol_mask,
            params["volume_shape"],
            params["res_s"] / params["res_r"],
            False,
            False,
        )

    def AtA(
        self,
        transforms: torch.Tensor,
        x: torch.Tensor,
        p: Optional[torch.Tensor],
        params: Dict,
        slices_mask: Optional[torch.Tensor] = None,
        vol_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        slices = self.A(transforms, x, params, slices_mask, vol_mask)
        if p is not None:
            slices = slices * p
        vol = self.At(transforms, slices, params, slices_mask, vol_mask)
        return vol
