#!/usr/bin/env bash
# Usage: source scripts/activate_lerobot.sh

_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_ENV_NAME="${LEROBOT_ENV_NAME:-lerobot-libero}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required; create the environment with: conda env create -f environment.yml" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$_ENV_NAME"

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$_ROOT/.libero}"
export LEROBOT_DATA_HOME="${LEROBOT_DATA_HOME:-$_ROOT/.cache/lerobot}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$_ROOT/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$_ROOT/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME/fontconfig"
cd "$_ROOT" || return 1 2>/dev/null || exit 1

unset _ROOT _ENV_NAME
