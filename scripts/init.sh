#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$BASE_DIR/env_tools/docker/isaaclab/.env"
ISAACLAB_DIR="$BASE_DIR/env_tools/docker/isaaclab/IsaacLab"
DEFAULT_ISAACLAB_REPO="https://github.com/isaac-sim/IsaacLab.git"
DEFAULT_ISAACLAB_VERSION="v2.3.2"

if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

ISAACLAB_REPO="${ISAACLAB_REPO:-$DEFAULT_ISAACLAB_REPO}"
ISAACLAB_VERSION="${ISAACLAB_VERSION:-$DEFAULT_ISAACLAB_VERSION}"

echo "Initializing IsaacLab source..."
echo "Repo: $ISAACLAB_REPO"
echo "Version: $ISAACLAB_VERSION"

# Keep the same initialization entry pattern as Crazy_Fast.
if git -C "$BASE_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "$BASE_DIR" submodule update --init --recursive || true
fi

if [ -d "$ISAACLAB_DIR/.git" ]; then
    git -C "$ISAACLAB_DIR" fetch --tags origin
    git -C "$ISAACLAB_DIR" checkout "$ISAACLAB_VERSION"
else
    rm -rf "$ISAACLAB_DIR"
    git clone --depth 1 --branch "$ISAACLAB_VERSION" "$ISAACLAB_REPO" "$ISAACLAB_DIR"
fi

git -C "$ISAACLAB_DIR" submodule update --init --recursive

mkdir -p "$BASE_DIR/logs"

echo "Initialized YOPO_isaac_lab scaffold and IsaacLab source."
