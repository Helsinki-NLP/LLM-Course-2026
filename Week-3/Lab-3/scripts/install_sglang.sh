#!/bin/bash
# Install SGLang into its own Python environment (several GB).
#
# For the course, the teacher runs this once on the GPU login node
# (ssh roihu-gpu.csc.fi), and everyone uses the shared environment:
#     bash scripts/install_sglang.sh /projappl/project_2021893/sglang-env
# Students only need it to install a private copy:
#     bash scripts/install_sglang.sh /scratch/project_2021893/$USER/sglang-env
#     export SGLANG_ENV=/scratch/project_2021893/$USER/sglang-env
set -euo pipefail

TARGET="${1:-/projappl/project_2021893/sglang-env}"
SGLANG_VERSION="${SGLANG_VERSION:-0.5.23}"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "Run this on the GPU side of Roihu (ssh roihu-gpu.csc.fi); this machine is $(uname -m)." >&2
  exit 2
fi

# uv provides a recent Python and installs the packages quickly.
python3 -m pip install --user --quiet uv
UV="$(python3 -m site --user-base)/bin/uv"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/project_2021893/${USER}/uv-cache}"

"$UV" venv --python 3.12 "$TARGET"
"$UV" pip install --python "$TARGET/bin/python" "sglang==${SGLANG_VERSION}"
chmod -R g+rX "$TARGET" 2>/dev/null || true

"$TARGET/bin/python" -c "import sglang, torch; print('SGLang', sglang.__version__, '| torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
echo "Installed SGLang in $TARGET"
