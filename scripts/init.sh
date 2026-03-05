#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$BASE_DIR/env_tools/docker/isaaclab/.env"
ISAACLAB_DIR="$BASE_DIR/env_tools/docker/isaaclab/IsaacLab"
DEFAULT_ISAACLAB_REPO="https://github.com/isaac-sim/IsaacLab.git"
DEFAULT_ISAACLAB_VERSION="v2.3.2"

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Error: required command not found: $1"
        exit 1
    fi
}

has_isaaclab_submodule_config() {
    local rel_path="env_tools/docker/isaaclab/IsaacLab"
    [ -f "$BASE_DIR/.gitmodules" ] || return 1
    git -C "$BASE_DIR" config -f .gitmodules --get-regexp '^submodule\..*\.path$' 2>/dev/null \
        | awk '{print $2}' | grep -Fxq "$rel_path"
}

if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

ISAACLAB_REPO="${ISAACLAB_REPO:-$DEFAULT_ISAACLAB_REPO}"
ISAACLAB_VERSION="${ISAACLAB_VERSION:-$DEFAULT_ISAACLAB_VERSION}"

require_command git

echo "Initializing IsaacLab source..."
echo "Repo: $ISAACLAB_REPO"
echo "Version: $ISAACLAB_VERSION"

# Keep the same initialization entry pattern as Crazy_Fast.
if git -C "$BASE_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if has_isaaclab_submodule_config; then
        git -C "$BASE_DIR" submodule update --init --recursive
    else
        echo "Skip root submodule update: no valid .gitmodules entry for IsaacLab."
    fi
fi

if [ -d "$ISAACLAB_DIR/.git" ]; then
    if git -C "$ISAACLAB_DIR" fetch --tags origin >/dev/null 2>&1; then
        echo "Fetched latest IsaacLab refs from origin."
    else
        echo "Warning: fetch from origin failed. Falling back to local IsaacLab refs."
    fi
    if ! git -C "$ISAACLAB_DIR" rev-parse --verify --quiet "${ISAACLAB_VERSION}^{commit}" >/dev/null; then
        echo "Error: IsaacLab version '$ISAACLAB_VERSION' is not available locally."
        echo "Please check ISAACLAB_VERSION/ISAACLAB_REPO or rerun init with network access."
        exit 1
    fi
    git -C "$ISAACLAB_DIR" checkout "$ISAACLAB_VERSION"
else
    rm -rf "$ISAACLAB_DIR"
    git clone --depth 1 --branch "$ISAACLAB_VERSION" "$ISAACLAB_REPO" "$ISAACLAB_DIR"
fi

git -C "$ISAACLAB_DIR" submodule update --init --recursive

if [ ! -x "$ISAACLAB_DIR/isaaclab.sh" ]; then
    echo "Error: init finished but '$ISAACLAB_DIR/isaaclab.sh' is missing."
    exit 1
fi

mkdir -p "$BASE_DIR/logs"

echo "Initialized YOPO_isaac_lab scaffold and IsaacLab source."
