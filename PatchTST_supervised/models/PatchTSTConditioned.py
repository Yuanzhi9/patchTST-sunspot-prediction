"""PatchTST with explicit FiLM conditioning on cycle-level variables."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp


class Model(nn.Module):
    def __init__(
        self,
        configs,
        max_seq_len: Optional[int] = 1024,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        act: str = "gelu",
        key_padding_mask: bool = "auto",
        padding_var: Optional[int] = None,
        attn_mask: Optional[Tensor] = None,
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        pretrain_head: bool = False,
        head_type: str = "flatten",
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        act = getattr(configs, "activation", act)
        self.c_in = int(configs.enc_in)
        self.cond_dim = int(getattr(configs, "cond_dim", 2))
        self.decomposition = int(configs.decomposition)

        backbone_kwargs = dict(
            c_in=self.c_in,
            context_window=configs.seq_len,
            target_window=configs.pred_len,
            patch_len=configs.patch_len,
            stride=configs.stride,
            max_seq_len=max_seq_len,
            n_layers=configs.e_layers,
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=configs.d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=configs.dropout,
            act=act,
            key_padding_mask=key_padding_mask,
            padding_var=padding_var,
            attn_mask=attn_mask,
            res_attention=res_attention,
            pre_norm=pre_norm,
            store_attn=store_attn,
            pe=pe,
            learn_pe=learn_pe,
            fc_dropout=configs.fc_dropout,
            head_dropout=configs.head_dropout,
            padding_patch=configs.padding_patch,
            pretrain_head=pretrain_head,
            head_type=head_type,
            individual=configs.individual,
            revin=configs.revin,
            affine=configs.affine,
            subtract_last=configs.subtract_last,
            verbose=verbose,
            **kwargs,
        )
        if self.decomposition:
            self.decomp_module = series_decomp(configs.kernel_size)
            self.model_trend = PatchTST_backbone(**backbone_kwargs)
            self.model_res = PatchTST_backbone(**backbone_kwargs)
        else:
            self.model = PatchTST_backbone(**backbone_kwargs)

        self.cond_mlp = nn.Sequential(
            nn.Linear(self.cond_dim, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, 2 * configs.d_model),
        )
        # Start as an exact no-condition model and let training learn modulation.
        nn.init.zeros_(self.cond_mlp[-1].weight)
        nn.init.zeros_(self.cond_mlp[-1].bias)

    @staticmethod
    def _apply_condition(backbone: PatchTST_backbone, z: Tensor, cond: Tensor, cond_mlp: nn.Module) -> Tensor:
        gamma_beta = cond_mlp(cond).view(cond.shape[0], 2, z.shape[2])
        gamma = gamma_beta[:, 0].view(-1, 1, z.shape[2], 1)
        beta = gamma_beta[:, 1].view(-1, 1, z.shape[2], 1)
        return z * (1.0 + gamma) + beta

    def _forward_backbone(self, backbone: PatchTST_backbone, x: Tensor, cond: Tensor) -> Tensor:
        z = x.permute(0, 2, 1)
        if backbone.revin:
            z = z.permute(0, 2, 1)
            z = backbone.revin_layer(z, "norm")
            z = z.permute(0, 2, 1)
        if backbone.padding_patch == "end":
            z = backbone.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=backbone.patch_len, step=backbone.stride)
        z = z.permute(0, 1, 3, 2)
        z = backbone.backbone(z)
        z = self._apply_condition(backbone, z, cond, self.cond_mlp)
        z = backbone.head(z)
        if backbone.revin:
            z = z.permute(0, 2, 1)
            z = backbone.revin_layer(z, "denorm")
            z = z.permute(0, 2, 1)
        return z

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        if cond.ndim == 1:
            cond = cond.unsqueeze(0)
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res = self._forward_backbone(
                self.model_res, res_init.permute(0, 2, 1), cond
            )
            trend = self._forward_backbone(
                self.model_trend, trend_init.permute(0, 2, 1), cond
            )
            return (res + trend).permute(0, 2, 1)
        return self._forward_backbone(self.model, x, cond).permute(0, 2, 1)
