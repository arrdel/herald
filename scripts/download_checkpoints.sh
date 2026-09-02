#!/usr/bin/env bash
# Fetch public checkpoints for the inference-only phase-0/1/2 experiments.
#
# Everything HERALD needs at this stage is a released checkpoint -- the method is training-free,
# so no ImageNet training run is required to get the headline numbers.
set -euo pipefail

cd "$(dirname "$0")/.."
CKPT_DIR=${CKPT_DIR:-checkpoints}
mkdir -p "$CKPT_DIR"/{vae,mar,djepa}

# hub >=0.34 renamed the CLI to `hf`; the old `huggingface-cli` entrypoint still resolves on PATH
# but exits non-zero with a deprecation notice, so probing with `command -v` is not enough.
if command -v hf >/dev/null 2>&1; then
  HF_DL=(hf download)
elif huggingface-cli --help >/dev/null 2>&1; then
  HF_DL=(huggingface-cli download)
else
  echo "error: no working Hugging Face CLI. Run: conda activate herald && pip install -e '.[dev]'" >&2
  exit 1
fi

echo "==> D-JEPA ImageNet checkpoints (B/L/H) -- includes the feature predictor and EMA target"
echo "    encoder that HERALD repurposes as drafter and verifier."
"${HF_DL[@]}" densechen/d-jepa-imagenet --local-dir "$CKPT_DIR/djepa"

echo
echo "==> MAR checkpoints -- common substrate, ablation floor, and CSpD's target model."
"${HF_DL[@]}" jadechoghari/mar --local-dir "$CKPT_DIR/mar" || {
  echo "    note: MAR weights may need the Google Drive links in baselines/mar/README.md" >&2
}

echo
echo "==> KL-16 VAE -- ships inside the MAR repo, linked into vae/ so both D-JEPA and CSpD find it"
echo "    at one canonical path regardless of which baseline is running."
if [ -f "$CKPT_DIR/mar/kl16.safetensors" ]; then
  ln -sf ../mar/kl16.safetensors "$CKPT_DIR/vae/kl16.safetensors"
  echo "    linked $CKPT_DIR/vae/kl16.safetensors"
else
  echo "    warning: kl16.safetensors not found under $CKPT_DIR/mar." >&2
  echo "    The MAR download above must have failed; E0.1 cannot run without it." >&2
fi

echo
echo "Checkpoint tree:"
du -sh "$CKPT_DIR"/* 2>/dev/null || true
