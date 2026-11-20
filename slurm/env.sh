# Sourced by every job: venv, no pip/uv caches in home, model and data folders in the project.
source ~/venvs/uplift/bin/activate
export UV_NO_CACHE=1 PIP_NO_CACHE_DIR=1
export UPLIFT_ROOT=~/TFM-uplift
export UPLIFT_DATA=$UPLIFT_ROOT/data
export HF_HOME=$UPLIFT_ROOT/models/hf
export TABPFN_MODEL_CACHE_DIR=$UPLIFT_ROOT/models/tabpfn
mkdir -p "$UPLIFT_DATA" "$HF_HOME" "$TABPFN_MODEL_CACHE_DIR"
# TabPFN licence key: kept outside the repo, in ~/.tabpfn_token (one line, chmod 600).
if [ -f ~/.tabpfn_token ]; then export TABPFN_TOKEN="$(cat ~/.tabpfn_token)"; fi
