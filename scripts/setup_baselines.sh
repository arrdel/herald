#!/usr/bin/env bash
# Clone upstream baseline repos into baselines/ (git-ignored).
# Keep these pristine -- never patch upstream in place. Anything HERALD needs to change goes in
# herald/models/ as an adapter, so `git -C baselines/<repo> diff` stays empty and reproductions
# remain trustworthy.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p baselines

clone() {
  local url=$1 dest=$2
  if [[ -d "baselines/$dest/.git" ]]; then
    echo "  [skip] $dest already cloned"
  else
    echo "  [clone] $dest"
    git clone --depth 1 "$url" "baselines/$dest"
  fi
}

echo "Cloning baselines..."
clone https://github.com/D-JEPA/djepa-imagenet.git djepa-imagenet   # target model + JEPA predictor
clone https://github.com/MarkXCloud/CSpD.git        CSpD            # direct competitor (E0.2, E0.3)
clone https://github.com/LTH14/mar.git              mar             # common substrate / ablation floor
clone https://github.com/soobin-um/jepa-guidance.git jepa-guidance  # JEPA density signal reference

echo
echo "Done. Verify they are unmodified with:"
echo "  for r in baselines/*/; do echo \"\$r\"; git -C \"\$r\" status --porcelain; done"
