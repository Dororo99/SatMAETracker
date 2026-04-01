"""
Satellite-aware BEVFormerBackbone for MapTracker.
Extends BEVFormerBackbone to pass satellite tokens through to encoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models import BACKBONES
from .bevformer_backbone import BEVFormerBackbone


@BACKBONES.register_module()
class SatBEVFormerBackbone(BEVFormerBackbone):

    def __init__(self, sat_grid_size=14, **kwargs):
        super().__init__(**kwargs)
        self.sat_grid_size = sat_grid_size

    def _get_sat_pos(self, bev_pos, sat_grid_size):
        """Interpolate BEV positional encoding to satellite grid size."""
        # bev_pos: (bs, C, bev_h, bev_w)
        sat_pos = F.interpolate(
            bev_pos, size=(sat_grid_size, sat_grid_size),
            mode='bilinear', align_corners=False)
        sat_pos = sat_pos.flatten(2).permute(0, 2, 1)  # (bs, num_sat, C)
        return sat_pos

    def forward(self, img, img_metas, timestep, history_bev_feats, history_img_metas,
                all_history_coord, *args, prev_bev=None, sat_tokens=None,
                sat_grid_size=None, img_backbone_gradient=True, **kwargs):
        from contextlib import nullcontext
        backprop_context = torch.no_grad if img_backbone_gradient is False else nullcontext
        with backprop_context():
            mlvl_feats = self.extract_img_feat(img=img, img_metas=img_metas)

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        bev_queries = self.bev_embedding.weight.to(dtype)

        # Temporal warping (same as BEVFormerBackbone)
        if len(history_bev_feats) > 0:
            all_warped_history_feat = []
            for b_i in range(bs):
                history_coord = all_history_coord[b_i]
                history_bev_feats_i = torch.stack(
                    [feats[b_i] for feats in history_bev_feats], 0)
                warped_history_feat_i = F.grid_sample(
                    history_bev_feats_i, history_coord,
                    padding_mode='zeros', align_corners=False)
                all_warped_history_feat.append(warped_history_feat_i)
            all_warped_history_feat = torch.stack(all_warped_history_feat, dim=0)
            prop_bev_feat = all_warped_history_feat[:, -1]
        else:
            all_warped_history_feat = None
            prop_bev_feat = None

        # Pad history buffer
        if len(history_bev_feats) < self.history_steps:
            num_repeat = self.history_steps - len(history_bev_feats)
            zero_bev_feats = torch.zeros(
                [bs, bev_queries.shape[1], self.bev_h, self.bev_w]).to(bev_queries.device)
            padding = torch.stack([zero_bev_feats] * num_repeat, dim=1)
            if all_warped_history_feat is not None:
                all_warped_history_feat = torch.cat(
                    [padding, all_warped_history_feat], dim=1)
            else:
                all_warped_history_feat = padding

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        # Generate satellite positional encoding
        sat_pos = None
        if sat_tokens is not None:
            grid_size = sat_grid_size or self.sat_grid_size
            sat_pos = self._get_sat_pos(bev_pos, grid_size)

        outs = self.transformer.get_bev_features(
            mlvl_feats,
            bev_queries,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h,
                         self.real_w / self.bev_w),
            bev_pos=bev_pos,
            prop_bev=prop_bev_feat,
            img_metas=img_metas,
            prev_bev=prev_bev,
            warped_history_bev=all_warped_history_feat,
            sat_tokens=sat_tokens,
            sat_pos=sat_pos,
        )

        outs = outs.unflatten(1, (self.bev_h, self.bev_w)).permute(0, 3, 1, 2).contiguous()

        if self.upsample:
            outs = self.up(outs)

        return outs, mlvl_feats
