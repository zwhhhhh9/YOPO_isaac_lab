#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

ISAACLAB_PATH="/workspace/isaaclab"
ISAAC_SIM_PATH="${ISAACLAB_PATH}/_isaac_sim"

check_environment() {
    [ -d "$ISAACLAB_PATH" ] || {
        echo "Error: IsaacLab not found at $ISAACLAB_PATH"
        exit 1
    }

    [ -L "$ISAAC_SIM_PATH" ] || [ -d "$ISAAC_SIM_PATH" ] || {
        echo "Error: Isaac Sim not found at $ISAAC_SIM_PATH"
        exit 1
    }

    [ -x "${ISAAC_SIM_PATH}/python.sh" ] || {
        echo "Error: Isaac Sim python.sh not executable at $ISAAC_SIM_PATH"
        exit 1
    }
}

setup_directories() {
    echo "Syncing project directories..."
    rsync -av --delete "${BASE_DIR}/yopo_drone/" "${ISAACLAB_PATH}/yopo_drone/"
    mkdir -p "${ISAACLAB_PATH}/logs"
}

setup_environment() {
    export ISAACSIM_PATH="$ISAAC_SIM_PATH"
    export OMNI_KIT_ALLOW_ROOT=1
    export QT_X11_NO_MITSHM=1

    local python_path="${PYTHONPATH:-}"
    [ -n "$python_path" ] && python_path="${python_path}:"
    export PYTHONPATH="${python_path}${ISAACLAB_PATH}"

    mkdir -p "${BASE_DIR}/logs"
}

main() {
    [ $# -eq 0 ] && {
        echo "Usage: $0 <script.py> [args...]"
        echo "Example: $0 yopo_drone/tasks/yopo/train.py --help"
        exit 1
    }

    check_environment
    setup_directories
    setup_environment

    cd "$ISAACLAB_PATH"
    exec "${ISAAC_SIM_PATH}/python.sh" "${ISAACLAB_PATH}/yopo_drone/run.py" "$@"
}

main "$@"
