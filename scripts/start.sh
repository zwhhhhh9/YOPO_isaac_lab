#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

if [ -n "${HOST_BASE_DIR:-}" ]; then
    echo "Running inside Docker, using HOST_BASE_DIR: $HOST_BASE_DIR"
    SCRIPT_DIR_NO_FIRST=$(echo "$(dirname "$SCRIPT_DIR")" | cut -d'/' -f3-)
    export BASE_DIR="$HOST_BASE_DIR/$SCRIPT_DIR_NO_FIRST"
else
    export BASE_DIR="$(dirname "$SCRIPT_DIR")"
    echo "Using local BASE_DIR: $BASE_DIR"
fi

TIMESTAMP=$(date +%Y%m%d%H%M%S)
PROJECT_NAME="yopo_isaaclab_${TIMESTAMP}"
DOCKER_COMPOSE_FILE="$(dirname "$SCRIPT_DIR")/env_tools/docker/isaaclab/docker-compose.yml"

# ---- NEW: default to no extra installs (no RL frameworks) ----
# Dockerfile / docker-compose should consume this variable.
export ISAACLAB_INSTALL_MODE="${ISAACLAB_INSTALL_MODE:-none}"
echo "ISAACLAB_INSTALL_MODE=${ISAACLAB_INSTALL_MODE}"

ensure_initialized() {
    local isaaclab_dir="$BASE_DIR/env_tools/docker/isaaclab/IsaacLab"
    if [ ! -x "$isaaclab_dir/isaaclab.sh" ]; then
        echo "IsaacLab source is missing or not initialized at: $isaaclab_dir"
        echo "Please run: ./scripts/init.sh"
        trap - EXIT INT TERM
        exit 1
    fi
}

create_shared_volumes() {
    echo "Setting up shared volumes..."

    SHARED_VOLUMES=(
        "isaac_shared_cache_kit"
        "isaac_shared_cache_ov"
        "isaac_shared_cache_pip"
        "isaac_shared_cache_gl"
        "isaac_shared_cache_compute"
        "isaac_shared_logs"
        "isaac_shared_carb_logs"
        "isaac_shared_data"
        "isaac_shared_docs"
        "isaac_shared_lab_data"
    )

    for VOLUME in "${SHARED_VOLUMES[@]}"; do
        if ! docker volume inspect "$VOLUME" &>/dev/null; then
            echo "Creating volume: $VOLUME"
            docker volume create "$VOLUME"
        fi
    done

    export ISAAC_CACHE_KIT=isaac_shared_cache_kit
    export ISAAC_CACHE_OV=isaac_shared_cache_ov
    export ISAAC_CACHE_PIP=isaac_shared_cache_pip
    export ISAAC_CACHE_GL=isaac_shared_cache_gl
    export ISAAC_CACHE_COMPUTE=isaac_shared_cache_compute
    export ISAAC_LOGS=isaac_shared_logs
    export ISAAC_CARB_LOGS=isaac_shared_carb_logs
    export ISAAC_DATA=isaac_shared_data
    export ISAAC_DOCS=isaac_shared_docs
    export ISAAC_LAB_DATA=isaac_shared_lab_data
}

cleanup() {
    echo "Cleaning up container: $PROJECT_NAME"
    docker compose -f "$DOCKER_COMPOSE_FILE" -p "$PROJECT_NAME" down
    echo "Container cleaned up successfully."
}

trap cleanup EXIT INT TERM

show_help() {
    echo "Usage: ./scripts/start.sh [options] [command]"
    echo ""
    echo "Before first run: ./scripts/init.sh"
    echo ""
    echo "Options:"
    echo "  -h, --help         Display this help message and exit"
    echo "  --stop-all         Stop all running YOPO Isaac Lab containers"
    echo ""
    echo "Env:"
    echo "  ISAACLAB_INSTALL_MODE=none   (default) Skip extra installs (RL frameworks)"
    echo ""
    echo "Example:"
    echo "  ISAACLAB_INSTALL_MODE=none ./scripts/start.sh yopo_drone/tasks/yopo/train.py --help"
    trap - EXIT INT TERM
    exit 0
}

stop_all_containers() {
    echo "Stopping all yopo_isaaclab containers..."
    PROJECTS=$(docker compose ls --format "{{.Project}}" | grep "^yopo_isaaclab_" || true)

    if [ -z "$PROJECTS" ]; then
        echo "No running yopo_isaaclab containers found."
    else
        for PROJECT in $PROJECTS; do
            echo "Stopping project: $PROJECT"
            docker compose -p "$PROJECT" down
        done
        echo "All yopo_isaaclab containers stopped."
    fi

    trap - EXIT INT TERM
    exit 0
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    show_help
elif [ "${1:-}" = "--stop-all" ]; then
    stop_all_containers
fi

ensure_initialized
create_shared_volumes

if [ "$#" -ge 1 ]; then
    ENTRYPOINT_CMD="/bin/bash -c \"source ~/.bashrc && /workspace/isaaclab/_isaac_sim/python.sh /workspace/isaaclab/yopo_drone/run.py $*\""
    export ENTRYPOINT="$ENTRYPOINT_CMD"
    echo "Setting ENTRYPOINT to: $ENTRYPOINT"
else
    echo "No ENTRYPOINT specified. Container will use default entrypoint."
fi

xhost + || true
mkdir -p "$BASE_DIR/logs"

echo "Starting new container with project name: $PROJECT_NAME"
docker compose -f "$DOCKER_COMPOSE_FILE" -p "$PROJECT_NAME" up