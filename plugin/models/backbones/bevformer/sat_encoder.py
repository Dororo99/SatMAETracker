"""
BEVFormer encoder and layer with Satellite Cross-Attention for MapTracker.

BEVFormerLayerWithSat:
  operation_order = ('self_attn', 'norm', 'cross_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
  attn_cfgs = [TemporalSelfAttn, MHA(satellite), SpatialCrossAttn]

BEVFormerEncoderWithSat:
  Extends BEVFormerEncoder with sat_tokens forwarding + TemporalNet fusion.
"""
import copy
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
from einops import rearrange

from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
from .encoder import BEVFormerEncoder
from .temporal_net import TemporalNet


@TRANSFORMER_LAYER.register_module()
class BEVFormerLayerWithSat(MyCustomBaseTransformerLayer):
    """BEVFormerLayer with additional Satellite Cross-Attention.

    Expects 3 attention modules in attn_cfgs:
      [0] TemporalSelfAttention (self_attn)
      [1] MultiheadAttention for satellite tokens (first cross_attn)
      [2] SpatialCrossAttention for camera features (second cross_attn)

    Satellite cross-attention output is gated (init=0 → no effect initially).
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 sat_gate_init=0.0,
                 **kwargs):
        super().__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        assert len(operation_order) == 8, \
            f'BEVFormerLayerWithSat expects 8 operations, got {len(operation_order)}'

        # Learnable gate: sigmoid(-3)≈0.05, starts near-zero, can only go [0,1]
        self.sat_gate = nn.Parameter(torch.tensor(float(sat_gate_init)))

    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                sat_tokens=None,
                sat_pos=None,
                **kwargs):
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        cross_attn_count = 0
        identity = query

        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]

        for layer in self.operation_order:
            if layer == 'self_attn':
                # Temporal Self-Attention (unchanged)
                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor(
                        [[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                if cross_attn_count == 0:
                    # === Satellite Cross-Attention (gated) ===
                    if sat_tokens is not None:
                        sat_out = self.attentions[attn_index](
                            query,
                            sat_tokens,
                            sat_tokens,
                            identity if self.pre_norm else None,
                            query_pos=bev_pos,
                            key_pos=sat_pos,
                            attn_mask=attn_masks[attn_index])
                        gate = torch.sigmoid(self.sat_gate)
                        query = (1 - gate) * query + gate * sat_out
                    attn_index += 1
                    cross_attn_count += 1
                    identity = query
                else:
                    # === Spatial Cross-Attention (unchanged) ===
                    query = self.attentions[attn_index](
                        query,
                        key,
                        value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=key_pos,
                        reference_points=ref_3d,
                        reference_points_cam=reference_points_cam,
                        mask=mask,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=key_padding_mask,
                        spatial_shapes=spatial_shapes,
                        level_start_index=level_start_index,
                        **kwargs)
                    attn_index += 1
                    cross_attn_count += 1
                    identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFormerEncoderWithSat(BEVFormerEncoder):
    """BEVFormer encoder with satellite tokens + TemporalNet (MapTracker).

    Same as BEVFormerEncoder but forwards sat_tokens/sat_pos to each layer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @auto_fp16()
    def forward(self,
                bev_query,
                key,
                value,
                *args,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                shift=0.,
                warped_history_bev=None,
                sat_tokens=None,
                sat_pos=None,
                **kwargs):

        output = bev_query
        intermediate = []

        ref_3d = self.get_reference_points(
            bev_h, bev_w, self.pc_range[5] - self.pc_range[2],
            self.num_points_in_pillar, dim='3d',
            bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)
        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim='2d',
            bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)

        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, kwargs['img_metas'])

        shift_ref_2d = ref_2d.clone()
        shift_ref_2d += shift[:, None, None, :]

        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape

        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            prev_bev = torch.stack(
                [prev_bev, bev_query], 1).reshape(bs * 2, len_bev, -1)
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs * 2, len_bev, num_bev_level, 2)
        else:
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                bs * 2, len_bev, num_bev_level, 2)

        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                warped_history_bev=warped_history_bev,
                sat_tokens=sat_tokens,
                sat_pos=sat_pos,
                **kwargs)

            # TemporalNet fusion (same as MapTracker)
            mem_layer = self.temporal_mem_layers[lid]
            curr_feat = rearrange(output, 'b (h w) c -> b c h w',
                                  h=warped_history_bev.shape[3])
            fused_output = mem_layer(warped_history_bev, curr_feat)
            fused_output = rearrange(fused_output, 'b c h w -> b (h w) c')
            output = output + fused_output

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output
