#!/usr/bin/env bash
# Source this file from any shell: source scripts/activate_lerobot.sh

_LEROBOT_PROJECT_ROOT="/workspace/jy/projects/lerobot-libero-benchmark"
_LEROBOT_CONDA_ROOT="/workspace/jy/miniforge3"
_LEROBOT_ENV_PREFIX="/workspace/jy/conda/envs/lerobot_libero"

if [[ ! -f "${_LEROBOT_CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    echo "Miniforge not found at ${_LEROBOT_CONDA_ROOT}" >&2
    return 1 2>/dev/null || exit 1
fi

# Avoid leaking ROS Python 3.10 packages or ~/.local packages into Python 3.12.
unset PYTHONPATH
export PYTHONNOUSERSITE=1

export HF_HOME="/workspace/jy/cache/huggingface"
export TORCH_HOME="/workspace/jy/cache/torch"
export PIP_CACHE_DIR="/workspace/jy/cache/pip"
export MUJOCO_GL="egl"
export LIBERO_CONFIG_PATH="${_LEROBOT_PROJECT_ROOT}/.libero"
export LEROBOT_DATA_HOME="/workspace/jy/datasets"
export LEROBOT_OUTPUT_HOME="/workspace/jy/checkpoints"
# Keep plotting/font caches inside this project instead of writing to ~/.cache.
export XDG_CACHE_HOME="${_LEROBOT_PROJECT_ROOT}/.cache"
export MPLCONFIGDIR="${_LEROBOT_PROJECT_ROOT}/.cache/matplotlib"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}/fontconfig"

# shellcheck disable=SC1091
source "${_LEROBOT_CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${_LEROBOT_ENV_PREFIX}"

# TorchCodec loads the ffmpeg libraries installed by Conda.  Putting the
# environment's lib directory first prevents the system libstdc++ from being
# mixed with Conda's newer ffmpeg/OpenVINO libraries.
case ":${LD_LIBRARY_PATH-}:" in
    *":${CONDA_PREFIX}/lib:"*) ;;
    *) export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
esac

cd "${_LEROBOT_PROJECT_ROOT}" || return 1 2>/dev/null || exit 1

unset _LEROBOT_PROJECT_ROOT _LEROBOT_CONDA_ROOT _LEROBOT_ENV_PREFIX
