#!/bin/bash
# Custom provisioning script for LTX 2.3 22B (distilled FP8) + Gemma 3 12B FP4
# Downloads the correct models for gerar_i2v_vastai.py pipeline
set -e

COMFYUI_DIR="${WORKSPACE:-/workspace}/ComfyUI"
CKPT_DIR="$COMFYUI_DIR/models/checkpoints"
CLIP_DIR="$COMFYUI_DIR/models/clip"

CHECKPOINT="ltx-2.3-22b-distilled-fp8.safetensors"
TEXT_ENCODER="gemma_3_12B_it_fp4_mixed.safetensors"

HF_AUTH=""
if [ -n "$HF_TOKEN" ]; then
    HF_AUTH="--header=Authorization: Bearer $HF_TOKEN"
fi

mkdir -p "$CKPT_DIR" "$CLIP_DIR"

# Download checkpoint (public repo)
if [ ! -f "$CKPT_DIR/$CHECKPOINT" ]; then
    echo "[1/2] Downloading $CHECKPOINT (~29.5GB)..."
    wget -q --show-progress -c \
        "https://huggingface.co/Lightricks/LTX-2.3-fp8/resolve/main/$CHECKPOINT" \
        -O "$CKPT_DIR/$CHECKPOINT"
else
    echo "[1/2] $CHECKPOINT already exists, skipping"
fi

# Download Gemma text encoder (gated repo, needs HF_TOKEN)
if [ ! -f "$CLIP_DIR/$TEXT_ENCODER" ]; then
    if [ -z "$HF_TOKEN" ]; then
        echo "[2/2] ERROR: HF_TOKEN required to download $TEXT_ENCODER from Comfy-Org/gemma (gated)"
        exit 1
    fi
    echo "[2/2] Downloading $TEXT_ENCODER (~6GB)..."
    wget -q --show-progress -c $HF_AUTH \
        "https://huggingface.co/Comfy-Org/gemma/resolve/main/$TEXT_ENCODER" \
        -O "$CLIP_DIR/$TEXT_ENCODER"
else
    echo "[2/2] $TEXT_ENCODER already exists, skipping"
fi

echo "Provisioning complete."
