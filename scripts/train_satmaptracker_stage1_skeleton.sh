#!/bin/bash
# SatMapTracker Stage1 BEV Pretrain with Skeleton-Recall Loss
#
# Usage:
#   bash scripts/train_satmaptracker_stage1_skeleton.sh                    # default (all ON)
#   bash scripts/train_satmaptracker_stage1_skeleton.sh --no-skeleton      # skeleton OFF
#   bash scripts/train_satmaptracker_stage1_skeleton.sh --no-cross-attn    # early fusion OFF
#   bash scripts/train_satmaptracker_stage1_skeleton.sh --no-conv-fusion   # late fusion OFF
#   bash scripts/train_satmaptracker_stage1_skeleton.sh --sat-gate -1.0    # cross-attn gate init
#   bash scripts/train_satmaptracker_stage1_skeleton.sh --fusion-gate -1.0 # conv fusion gate init
#   bash scripts/train_satmaptracker_stage1_skeleton.sh --gpus 4,5,6,7     # custom GPUs
#   bash scripts/train_satmaptracker_stage1_skeleton.sh --wandb-name my_exp # custom wandb name

set -e

# ============================================================
# Default settings
# ============================================================
GPUS="0,1"
NUM_GPUS=2
MASTER_PORT=29571
USE_SKELETON=true
SKEL_WEIGHT=1.0
SKEL_CLASSES="[1,2]"
USE_CROSS_ATTN=true
SAT_FUSION_MODE=gate
SAT_GATE_INIT=-1.0
USE_CONV_FUSION=true
FUSION_GATE_INIT=-1.0
BEV_VIS_INTERVAL=500
WANDB_ENTITY="IRCV_Mapping"
WANDB_PROJECT="Third-SatMAE_MapTracker-AID4AD-seonghyun"
WANDB_NAME="SatMAETracker_sig25_skeleton"
CONFIG="plugin/configs/maptracker/nuscenes_newsplit/satmaptracker_stage1_bev_pretrain.py"

# ============================================================
# Parse arguments
# ============================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPUS="$2"
            NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
            shift 2 ;;
        --port)
            MASTER_PORT="$2"
            shift 2 ;;
        --no-skeleton)
            USE_SKELETON=false
            shift ;;
        --skel-weight)
            SKEL_WEIGHT="$2"
            shift 2 ;;
        --no-cross-attn)
            USE_CROSS_ATTN=false
            shift ;;
        --sat-fusion-mode)
            SAT_FUSION_MODE="$2"
            shift 2 ;;
        --sat-gate)
            SAT_GATE_INIT="$2"
            shift 2 ;;
        --no-conv-fusion)
            USE_CONV_FUSION=false
            shift ;;
        --fusion-gate)
            FUSION_GATE_INIT="$2"
            shift 2 ;;
        --bev-vis-interval)
            BEV_VIS_INTERVAL="$2"
            shift 2 ;;
        --wandb-name)
            WANDB_NAME="$2"
            shift 2 ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2 ;;
        --config)
            CONFIG="$2"
            shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1 ;;
    esac
done

# ============================================================
# Build cfg-options string
# ============================================================
CFG_OPTIONS=""

# Skeleton loss control
if [ "$USE_SKELETON" = true ]; then
    CFG_OPTIONS+=" --cfg-options model.seg_cfg.loss_skel.type=SkelRecallLoss"
    CFG_OPTIONS+=" --cfg-options model.seg_cfg.loss_skel.loss_weight=${SKEL_WEIGHT}"
    CFG_OPTIONS+=" --cfg-options model.seg_cfg.loss_skel.skel_classes=${SKEL_CLASSES}"
    echo "[Config] Skeleton-Recall Loss: ON (weight=${SKEL_WEIGHT}, classes=${SKEL_CLASSES})"
else
    CFG_OPTIONS+=" --cfg-options model.seg_cfg.loss_skel=None"
    echo "[Config] Skeleton-Recall Loss: OFF"
fi

# Cross-attention (early fusion) control
if [ "$USE_CROSS_ATTN" = true ]; then
    CFG_OPTIONS+=" --cfg-options model.backbone_cfg.transformer.encoder.transformerlayers.sat_fusion_mode=${SAT_FUSION_MODE}"
    if [ "$SAT_FUSION_MODE" = "gate" ]; then
        CFG_OPTIONS+=" --cfg-options model.backbone_cfg.transformer.encoder.transformerlayers.sat_gate_init=${SAT_GATE_INIT}"
        echo "[Config] Cross-Attention (Early Fusion): ON (mode=gate, init=${SAT_GATE_INIT}, sigmoid≈$(python3 -c "import math; print(f'{1/(1+math.exp(-${SAT_GATE_INIT})):.0%}')"))"
    else
        echo "[Config] Cross-Attention (Early Fusion): ON (mode=add)"
    fi
else
    CFG_OPTIONS+=" --cfg-options model.sat_encoder_cfg=None"
    echo "[Config] Cross-Attention (Early Fusion): OFF"
fi

# Conv fusion (late fusion) control
if [ "$USE_CONV_FUSION" = true ]; then
    CFG_OPTIONS+=" --cfg-options model.conv_fusion_cfg.gate_init=${FUSION_GATE_INIT}"
    echo "[Config] Conv Fusion (Late Fusion): ON (gate_init=${FUSION_GATE_INIT}, sigmoid≈$(python3 -c "import math; print(f'{1/(1+math.exp(-${FUSION_GATE_INIT})):.0%}')"))"
else
    CFG_OPTIONS+=" --cfg-options model.conv_fusion_cfg=None"
    echo "[Config] Conv Fusion (Late Fusion): OFF"
fi

# WandB settings
CFG_OPTIONS+=" --cfg-options log_config.hooks.1.init_kwargs.entity=${WANDB_ENTITY}"
CFG_OPTIONS+=" --cfg-options log_config.hooks.1.init_kwargs.project=${WANDB_PROJECT}"
CFG_OPTIONS+=" --cfg-options log_config.hooks.1.init_kwargs.name=${WANDB_NAME}"

# BEV visualization interval
CFG_OPTIONS+=" --cfg-options custom_hooks.0.interval=${BEV_VIS_INTERVAL}"

# ============================================================
# Print settings
# ============================================================
echo "============================================"
echo " SatMapTracker Stage1 BEV Pretrain"
echo "============================================"
echo " GPUs:          ${GPUS} (${NUM_GPUS} devices)"
echo " Master Port:   ${MASTER_PORT}"
echo " Config:        ${CONFIG}"
echo " Cross-Attn:    ${USE_CROSS_ATTN} (mode=${SAT_FUSION_MODE}, gate=${SAT_GATE_INIT})"
echo " Conv Fusion:   ${USE_CONV_FUSION} (gate=${FUSION_GATE_INIT})"
echo " Skeleton:      ${USE_SKELETON} (weight=${SKEL_WEIGHT})"
echo " WandB:         ${WANDB_PROJECT} / ${WANDB_NAME}"
echo " BEV Vis:       every ${BEV_VIS_INTERVAL} iters"
echo "============================================"
echo ""

# ============================================================
# Run training
# ============================================================
export CUDA_VISIBLE_DEVICES=${GPUS}
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning"

cd $(dirname $(dirname $(realpath $0)))

/venv/maptracker/bin/python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    tools/train.py \
    ${CONFIG} \
    --launcher pytorch \
    ${CFG_OPTIONS}
