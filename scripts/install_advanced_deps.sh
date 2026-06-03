#!/usr/bin/env bash
# Install PyTorch + Depth Anything 3 on macOS without building xformers.
# xformers is optional in DA3 (SwiGLU falls back to pure PyTorch).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
PIP="${PIP:-$ROOT/.venv/bin/pip}"

if [[ ! -x "$PIP" ]]; then
  echo "Create venv first: python3.11 -m venv .venv && source .venv/bin/activate"
  exit 1
fi

echo "==> PyTorch (MPS/CUDA/CPU)"
"$PIP" install --upgrade pip
"$PIP" install "torch>=2.1" "torchvision>=0.16"

echo "==> Depth Anything 3 (no deps — skips xformers)"
"$PIP" install "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git" --no-deps

echo "==> DA3 runtime dependencies (no xformers, no open3d)"
"$PIP" install \
  "numpy<2" \
  einops huggingface_hub imageio "opencv-python>=4.9" pillow omegaconf safetensors \
  typer requests trimesh e3nn addict evo "moviepy==1.0.3" plyfile pycolmap

echo "==> Verify import"
export KMP_DUPLICATE_LIB_OK=TRUE
"$PYTHON" -c "from depth_anything_3.api import DepthAnything3; print('Depth Anything 3 OK')"

echo ""
echo "Done. For fishnet advanced runs, use:"
echo "  export KMP_DUPLICATE_LIB_OK=TRUE"
echo "  export FISHNET_DEPTH_MODEL=depth-anything/DA3-SMALL  # default; use DA3-BASE if you have disk space"
echo "  python main.py --pipeline advanced --method skeleton --grid-auto --depth --3d --limit 2 ..."
