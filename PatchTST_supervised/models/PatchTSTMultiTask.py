"""PatchTST with optional cycle-event auxiliary heads.

The curve output uses the same PatchTST backbone and flatten head as the
frozen baseline. Auxiliary heads read only the SSN channel's mean-pooled
patch representation.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp


MT_MODES = {
    "none": [],
    "amp": ["next_peak_raw"],
    "peak_time": ["months_to_next_peak"],
    "trough_time": ["months_to_next_trough"],
    "all": [
        "next_peak_raw",
        "months_to_next_peak",
        "months_to_next_trough",
    ],
}


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
        self.decomposition = int(configs.decomposition)
        self.mt_mode = str(getattr(configs, "mt_mode", "none"))
        self.mt_weighting = str(getattr(configs, "mt_weighting", "fixed"))
        if self.mt_mode not in MT_MODES:
            raise ValueError(f"unsupported mt_mode: {self.mt_mode}")
        if self.mt_weighting not in {"fixed", "uncertainty"}:
            raise ValueError(f"unsupported mt_weighting: {self.mt_weighting}")
        if self.decomposition:
            raise ValueError("PatchTSTMultiTask currently requires decomposition=0")
        if int(configs.individual):
            raise ValueError("PatchTSTMultiTask currently requires individual=0")

        backbone_kwargs = dict(
            c_in=configs.enc_in,
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
        self.model = PatchTST_backbone(**backbone_kwargs)
        self.aux_names = MT_MODES[self.mt_mode]
        self.aux_head = (
            nn.Linear(configs.d_model, len(self.aux_names))
            if self.aux_names
            else None
        )
        self.mt_log_vars = (
            nn.Parameter(torch.zeros(len(self.aux_names)))
            if self.aux_names and self.mt_weighting == "uncertainty"
            else None
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        model = self.model
        z = x.permute(0, 2, 1)
        if model.revin:
            z = z.permute(0, 2, 1)
            z = model.revin_layer(z, "norm")
            z = z.permute(0, 2, 1)
        if model.padding_patch == "end":
            z = model.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=model.patch_len, step=model.stride)
        z = z.permute(0, 1, 3, 2)
        representation = model.backbone(z)

        curve = model.head(representation)
        if model.revin:
            curve = curve.permute(0, 2, 1)
            curve = model.revin_layer(curve, "denorm")
            curve = curve.permute(0, 2, 1)
        curve = curve.permute(0, 2, 1)

        if self.aux_head is None:
            aux = curve.new_zeros((curve.shape[0], 4))
        else:
            # Feature order is month_sin, month_cos, ssn; use only SSN.
            ssn_representation = representation[:, -1, :, :].mean(dim=-1)
            raw_aux = self.aux_head(ssn_representation)
            aux = raw_aux.new_zeros((curve.shape[0], 4))
            column_map = {
                "next_peak_raw": 0,
                "months_to_next_peak": 1,
                "months_to_next_trough": 2,
            }
            for output_index, name in enumerate(self.aux_names):
                aux[:, column_map[name]] = raw_aux[:, output_index]
        return curve, aux
