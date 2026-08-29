#!/usr/bin/env bash
# Fetch public checkpoints for the inference-only phase-0/1/2 experiments.
#
# Everything HERALD needs at this stage is a released checkpoint -- the method is training-free,
# so no ImageNet training run is required to get the headline numbers.
set -euo pipefail

cd "$(dirname "$0")/.."
CKPT_DIR=${CKPT_DIR:-checkpoints}
mkdir -p "$CKPT_DIR"/{vae,mar,djepa}

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "error: huggingface-cli not found. Run: conda activate herald && pip install -e '.[dev]'" >&2
  exit 1
fi

echo "==> D-JEPA ImageNet checkpoints (B/L/H) -- includes the feature predictor and EMA target"
echo "    encoder that HERALD repurposes as drafter and verifier."
huggingface-cli download densechen/d-jepa-imagenet --local-dir "$CKPT_DIR/djepa"

echo
echo "==> MAR checkpoints -- common substrate, ablation floor, and CSpD's target model."
huggingface-cli download jadechoghari/mar --local-dir "$CKPT_DIR/mar" || {
  echo "    note: MAR weights may need the Google Drive links in baselines/mar/README.md" >&2
}

echo
echo "==> KL-16 VAE"
echo "    Not on the Hub. Download manually from the link in baselines/mar/README.md and place at:"
echo "      $CKPT_DIR/vae/kl16.ckpt"
echo "    Both D-JEPA and CSpD expect this exact file."

echo
echo "Checkpoint tree:"
du -sh "$CKPT_DIR"/* 2>/dev/null || true
