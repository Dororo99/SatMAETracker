"""
SatMapTracker: MapTracker with Satellite Cross-Attention in BEVFormer encoder.

Extends MapTracker to:
  1. Load satellite images via SatMAE encoder (frozen ViT-L)
  2. Pass satellite tokens to SatBEVFormerBackbone for cross-attention
  3. Everything else (TemporalNet, seg head, vector head) unchanged
"""
import torch
import torch.nn as nn

from mmdet3d.models.builder import build_backbone
from .base_mapper import MAPPERS
from .MapTracker import MapTracker


@MAPPERS.register_module()
class SatMapTracker(MapTracker):

    def __init__(self,
                 sat_encoder_cfg=None,
                 **kwargs):
        super().__init__(**kwargs)

        if sat_encoder_cfg is not None:
            self.sat_encoder = build_backbone(sat_encoder_cfg)
        else:
            self.sat_encoder = None

    def _encode_satellite(self, sat_img):
        if self.sat_encoder is None or sat_img is None:
            return None, None
        sat_tokens, grid_size = self.sat_encoder(sat_img)
        return sat_tokens, grid_size

    def forward_train(self, img, vectors, semantic_mask, points=None,
                      img_metas=None, all_prev_data=None,
                      all_local2global_info=None, sat_img=None, **kwargs):

        gts, img, img_metas, valid_idx, points = self.batch_data(
            vectors, img, img_metas, img.device, points)
        bs = img.shape[0]

        _use_memory = self.use_memory and self.num_iter > self.mem_warmup_iters

        if all_prev_data is not None:
            num_prev_frames = len(all_prev_data)
            all_gts_prev, all_img_prev, all_img_metas_prev, all_semantic_mask_prev = [], [], [], []
            all_sat_img_prev = []
            for prev_data in all_prev_data:
                gts_prev, img_prev, img_metas_prev, valid_idx_prev, _ = self.batch_data(
                    prev_data['vectors'], prev_data['img'], prev_data['img_metas'], img.device)
                all_gts_prev.append(gts_prev)
                all_img_prev.append(img_prev)
                all_img_metas_prev.append(img_metas_prev)
                all_semantic_mask_prev.append(prev_data['semantic_mask'])
                all_sat_img_prev.append(prev_data.get('sat_img', None))
        else:
            num_prev_frames = 0

        assert points is None

        if self.skip_vector_head:
            backprop_backbone_ids = [0, num_prev_frames]
        else:
            backprop_backbone_ids = [num_prev_frames]

        track_query_info = None
        all_loss_dict_prev = []
        all_trans_loss = []
        all_outputs_prev = []
        self.tracked_query_length = {}

        if _use_memory:
            self.memory_bank.set_bank_size(self.mem_len)
            self.memory_bank.init_memory(bs=bs)

        history_bev_feats = []
        history_img_metas = []
        gt_semantic = torch.flip(semantic_mask, [2,])

        # Iterate through prev frames
        for t in range(num_prev_frames):
            img_backbone_gradient = (t in backprop_backbone_ids)

            all_history_curr2prev, all_history_prev2curr, all_history_coord = \
                self.process_history_info(all_img_metas_prev[t], history_img_metas)

            # Encode satellite for this frame
            sat_img_t = all_sat_img_prev[t] if all_prev_data is not None else None
            sat_tokens_t, sat_grid_t = self._encode_satellite(sat_img_t)

            _bev_feats, mlvl_feats = self.backbone(
                all_img_prev[t], all_img_metas_prev[t], t,
                history_bev_feats, history_img_metas, all_history_coord,
                points=None, img_backbone_gradient=img_backbone_gradient,
                sat_tokens=sat_tokens_t, sat_grid_size=sat_grid_t)

            bev_feats = self.neck(_bev_feats)

            if _use_memory:
                self.memory_bank.curr_t = t

            if self.skip_vector_head or t == 0:
                self.temporal_propagate(bev_feats, all_img_metas_prev[t],
                    all_history_curr2prev, all_history_prev2curr,
                    _use_memory, track_query_info, timestep=t, get_trans_loss=False)
            else:
                trans_loss_dict = self.temporal_propagate(bev_feats, all_img_metas_prev[t],
                    all_history_curr2prev, all_history_prev2curr,
                    _use_memory, track_query_info, timestep=t, get_trans_loss=True)

            img_metas_prev = all_img_metas_prev[t]
            img_metas_next = all_img_metas_prev[t+1] if t < num_prev_frames-1 else img_metas
            gts_prev = all_gts_prev[t]
            gts_next = all_gts_prev[t+1] if t != num_prev_frames-1 else gts
            gts_semantic_prev = torch.flip(all_semantic_mask_prev[t], [2,])
            gts_semantic_curr = torch.flip(all_semantic_mask_prev[t+1], [2,]) if t != num_prev_frames-1 else gt_semantic

            local2global_prev = all_local2global_info[t]
            local2global_next = all_local2global_info[t+1]

            seg_preds, seg_feats, seg_loss, seg_dice_loss = self.seg_decoder(
                bev_feats, gts_semantic_prev, all_history_coord, return_loss=True)

            history_bev_feats.append(bev_feats)
            history_img_metas.append(all_img_metas_prev[t])
            if len(history_bev_feats) > self.history_steps:
                history_bev_feats.pop(0)
                history_img_metas.pop(0)

            if not self.skip_vector_head:
                gt_cur2prev, gt_prev2cur = self.get_two_frame_matching(
                    local2global_prev, local2global_next, gts_prev, gts_next)
                if t == 0:
                    memory_bank = None
                else:
                    memory_bank = self.memory_bank if _use_memory else None
                loss_dict_prev, outputs_prev, prev_inds_list, prev_gt_inds_list, \
                    prev_matched_reg_cost, prev_gt_list = self.head(
                        bev_features=bev_feats, img_metas=img_metas_prev,
                        gts=gts_prev, track_query_info=track_query_info,
                        memory_bank=memory_bank, return_loss=True, return_matching=True)
                all_outputs_prev.append(outputs_prev)
                if t > 0:
                    all_trans_loss.append(trans_loss_dict)

                pos_th = 0.4
                track_query_info = self.prepare_track_queries_and_targets(
                    gts_next, prev_inds_list, prev_gt_inds_list, prev_matched_reg_cost,
                    prev_gt_list, outputs_prev, gt_cur2prev, gt_prev2cur,
                    img_metas_prev, _use_memory, pos_th=pos_th, timestep=t)
            else:
                loss_dict_prev = {}

            loss_dict_prev['seg'] = seg_loss
            loss_dict_prev['seg_dice'] = seg_dice_loss
            all_loss_dict_prev.append(loss_dict_prev)

        if _use_memory:
            self.memory_bank.curr_t = num_prev_frames

        # Current frame
        img_backbone_gradient = num_prev_frames in backprop_backbone_ids

        all_history_curr2prev, all_history_prev2curr, all_history_coord = \
            self.process_history_info(img_metas, history_img_metas)

        # Encode satellite for current frame
        sat_tokens, sat_grid = self._encode_satellite(sat_img)

        _bev_feats, mlvl_feats = self.backbone(
            img, img_metas, num_prev_frames, history_bev_feats, history_img_metas,
            all_history_coord, points=None, img_backbone_gradient=img_backbone_gradient,
            sat_tokens=sat_tokens, sat_grid_size=sat_grid)

        bev_feats = self.neck(_bev_feats)

        if self.skip_vector_head or num_prev_frames == 0:
            assert track_query_info is None
            self.temporal_propagate(bev_feats, img_metas, all_history_curr2prev,
                all_history_prev2curr, _use_memory, track_query_info,
                timestep=num_prev_frames, get_trans_loss=False)
        else:
            trans_loss_dict = self.temporal_propagate(bev_feats, img_metas,
                all_history_curr2prev, all_history_prev2curr, _use_memory,
                track_query_info, timestep=num_prev_frames, get_trans_loss=True)
            all_trans_loss.append(trans_loss_dict)

        seg_preds, seg_feats, seg_loss, seg_dice_loss = self.seg_decoder(
            bev_feats, gt_semantic, all_history_coord, return_loss=True)

        if not self.skip_vector_head:
            memory_bank = self.memory_bank if _use_memory else None
            preds_list, loss_dict, det_match_idxs, det_match_gt_idxs, gt_list = self.head(
                bev_features=bev_feats, img_metas=img_metas, gts=gts,
                track_query_info=track_query_info, memory_bank=memory_bank,
                return_loss=True)
        else:
            loss_dict = {}

        loss_dict['seg'] = seg_loss
        loss_dict['seg_dice'] = seg_dice_loss

        # Aggregate losses
        loss = 0
        losses_t = []
        for loss_dict_t in (all_loss_dict_prev + [loss_dict]):
            loss_t = 0
            for name, var in loss_dict_t.items():
                loss_t = loss_t + var
            losses_t.append(loss_t)
            loss += loss_t

        for trans_loss_dict_t in all_trans_loss:
            trans_loss_t = trans_loss_dict_t['f_trans'] + trans_loss_dict_t['b_trans']
            loss += trans_loss_t

        log_vars = {k: v.item() for k, v in loss_dict.items()}
        for t, loss_dict_t in enumerate(all_loss_dict_prev):
            log_vars_t = {k+'_t{}'.format(t): v.item() for k, v in loss_dict_t.items()}
            log_vars.update(log_vars_t)
        for t, loss_t in enumerate(losses_t):
            log_vars.update({'total_t{}'.format(t): loss_t.item()})
        for t, trans_loss_dict_t in enumerate(all_trans_loss):
            log_vars_t = {k+'_t{}'.format(t): v.item() for k, v in trans_loss_dict_t.items()}
            log_vars.update(log_vars_t)
        log_vars.update({'total': loss.item()})

        # Log satellite gate values
        if hasattr(self.backbone, 'transformer'):
            encoder = self.backbone.transformer.encoder
            for lid, layer in enumerate(encoder.layers):
                if hasattr(layer, 'sat_gate'):
                    log_vars[f'sat_gate_L{lid}'] = torch.tanh(layer.sat_gate).item()

        num_sample = img.size(0)
        return loss, log_vars, num_sample

    @torch.no_grad()
    def forward_test(self, img, points=None, img_metas=None, seq_info=None,
                     sat_img=None, **kwargs):
        assert img.shape[0] == 1

        tokens = []
        for img_meta in img_metas:
            tokens.append(img_meta['token'])

        scene_name, local_idx, seq_length = seq_info[0]
        first_frame = (local_idx == 0)
        img_metas[0]['local_idx'] = local_idx

        if first_frame:
            if self.use_memory:
                self.memory_bank.set_bank_size(self.test_time_history_steps)
                self.memory_bank.init_memory(bs=1)
            self.history_bev_feats_all = []
            self.history_img_metas_all = []

        if self.use_memory:
            self.memory_bank.curr_t = local_idx

        selected_mem_ids = self.select_memory_entries(self.history_img_metas_all, img_metas)
        history_img_metas = [self.history_img_metas_all[idx] for idx in selected_mem_ids]
        history_bev_feats = [self.history_bev_feats_all[idx] for idx in selected_mem_ids]

        all_history_curr2prev, all_history_prev2curr, all_history_coord = \
            self.process_history_info(img_metas, history_img_metas)

        # Encode satellite
        sat_tokens, sat_grid = self._encode_satellite(sat_img)

        _bev_feats, mlvl_feats = self.backbone(
            img, img_metas, local_idx, history_bev_feats, history_img_metas,
            all_history_coord, points=points,
            sat_tokens=sat_tokens, sat_grid_size=sat_grid)

        bev_feats = self.neck(_bev_feats)

        if self.skip_vector_head or first_frame:
            self.temporal_propagate(bev_feats, img_metas, all_history_curr2prev,
                all_history_prev2curr, self.use_memory, track_query_info=None)
            seg_preds, seg_feats = self.seg_decoder(bev_features=bev_feats, return_loss=False)
            if not self.skip_vector_head:
                preds_list = self.head(bev_feats, img_metas=img_metas, return_loss=False)
            track_dict = None
        else:
            track_query_info = self.head.get_track_info(scene_name, local_idx)
            self.temporal_propagate(bev_feats, img_metas, all_history_curr2prev,
                all_history_prev2curr, self.use_memory, track_query_info)
            seg_preds, seg_feats = self.seg_decoder(bev_features=bev_feats, return_loss=False)
            memory_bank = self.memory_bank if self.use_memory else None
            preds_list = self.head(bev_feats, img_metas=img_metas,
                track_query_info=track_query_info, memory_bank=memory_bank,
                return_loss=False)
            track_dict = self._process_track_query_info(track_query_info)

        if not self.skip_vector_head:
            preds_dict = preds_list[-1]
        else:
            preds_dict = None

        self.history_bev_feats_all.append(bev_feats)
        self.history_img_metas_all.append(img_metas)
        if len(self.history_bev_feats_all) > self.test_time_history_steps:
            self.history_bev_feats_all.pop(0)
            self.history_img_metas_all.pop(0)

        if not self.skip_vector_head:
            memory_bank = self.memory_bank if self.use_memory else None
            thr_det = 0.4 if first_frame else 0.6
            pos_results = self.head.prepare_temporal_propagation(
                preds_dict, scene_name, local_idx, memory_bank,
                thr_track=0.5, thr_det=thr_det)

        if not self.skip_vector_head:
            results_list = self.head.post_process(preds_dict, tokens, track_dict)
            results_list[0]['pos_results'] = pos_results
            results_list[0]['meta'] = img_metas[0]
        else:
            results_list = [{'vectors': [], 'scores': [], 'labels': [],
                             'props': [], 'token': token} for token in tokens]

        for b_i in range(len(results_list)):
            tmp_scores, tmp_labels = seg_preds[b_i].max(0)
            tmp_scores = tmp_scores.sigmoid()
            preds_i = torch.zeros(tmp_labels.shape, dtype=torch.uint8).to(tmp_scores.device)
            pos_ids = tmp_scores >= 0.4
            preds_i[pos_ids] = tmp_labels[pos_ids].type(torch.uint8) + 1
            preds_i = preds_i.cpu().numpy()
            results_list[b_i]['semantic_mask'] = preds_i
            if 'token' not in results_list[b_i]:
                results_list[b_i]['token'] = tokens[b_i]

        return results_list
