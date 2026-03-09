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

if [ -z "${TERM:-}" ] || [ "${TERM:-}" = "dumb" ]; then
    export TERM="xterm-256color"
fi

TIMESTAMP=$(date +%Y%m%d%H%M%S)
PROJECT_NAME="yopo_isaaclab_${TIMESTAMP}"
DOCKER_COMPOSE_FILE="$(dirname "$SCRIPT_DIR")/env_tools/docker/isaaclab/docker-compose.yml"

export ISAACLAB_INSTALL_MODE="${ISAACLAB_INSTALL_MODE:-none}"
echo "ISAACLAB_INSTALL_MODE=${ISAACLAB_INSTALL_MODE}"

COMPOSE_STARTED=0

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Error: required command not found: $1"
        trap - EXIT INT TERM
        exit 1
    fi
}

ensure_docker_ready() {
    require_command docker
    if ! docker compose version >/dev/null 2>&1; then
        echo "Error: docker compose plugin is not available."
        trap - EXIT INT TERM
        exit 1
    fi
    if ! docker info >/dev/null 2>&1; then
        echo "Error: cannot access Docker daemon. Check docker service and user permissions."
        trap - EXIT INT TERM
        exit 1
    fi
}

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
    trap - EXIT INT TERM
    if [ "$COMPOSE_STARTED" -eq 1 ]; then
        echo "Cleaning up container: $PROJECT_NAME"
        docker compose -f "$DOCKER_COMPOSE_FILE" -p "$PROJECT_NAME" down || true
        echo "Container cleaned up successfully."
        COMPOSE_STARTED=0
    fi
}

trap cleanup EXIT INT TERM

show_help() {
    echo "Usage: ./scripts/start.sh [options] [command]"
    echo ""
    echo "Before first run: ./scripts/init.sh"
    echo ""
    echo "Options:"
    echo "  -h, --help        Display this help message and exit"
    echo "  --gui             Start Isaac Lab GUI directly"
    echo "  --env_editor      Run yopo_drone/env/drone_env_editor.py (GUI preview by default)"
    echo "  --ros2-node       Run a ROS 2 Python node with system Python inside the container"
    echo "  --stop-all        Stop all running YOPO Isaac Lab containers"
    echo ""
    echo "Env:"
    echo "  ISAACLAB_INSTALL_MODE=none   (default) Skip extra installs (RL frameworks)"
    echo "  ROS_DISTRO=jazzy             ROS 2 distro used for --ros2-node and sidecar setup"
    echo ""
    echo "Example:"
    echo "  ./scripts/start.sh --gui"
    echo "  ./scripts/start.sh --env_editor --help"
    echo "  ./scripts/start.sh --ros2-node yopo_drone/tasks/hover_initial_position.py"
    trap - EXIT INT TERM
    exit 0
}

stop_all_containers() {
    ensure_docker_ready
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

LAUNCH_GUI=0
LAUNCH_EDITOR=0
LAUNCH_ROS2_NODE=0

ROS_DISTRO="${ROS_DISTRO:-jazzy}"
ROS_WS="/workspace/isaaclab/ros2_ws"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
ROS_PREPARE_CMD="if [ -d ${ROS_WS}/src ]; then source /opt/ros/${ROS_DISTRO}/setup.bash >/dev/null 2>&1; if [ ! -f ${ROS_WS}/install/setup.bash ]; then cd ${ROS_WS} && colcon build --merge-install --symlink-install; fi; for d in ${ROS_WS}/build ${ROS_WS}/install ${ROS_WS}/log; do [ -e "\$d" ] && chown -R ${HOST_UID}:${HOST_GID} "\$d" || true; done; cd /workspace/isaaclab; fi;"
ROS_SOURCE_CMD="source /opt/ros/${ROS_DISTRO}/setup.bash >/dev/null 2>&1; [ -f ${ROS_WS}/install/setup.bash ] && source ${ROS_WS}/install/setup.bash >/dev/null 2>&1 || true;"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    show_help
elif [ "${1:-}" = "--stop-all" ]; then
    stop_all_containers
elif [ "${1:-}" = "--gui" ]; then
    LAUNCH_GUI=1
    shift
elif [ "${1:-}" = "--env_editor" ]; then
    LAUNCH_EDITOR=1
    shift
elif [ "${1:-}" = "--ros2-node" ]; then
    LAUNCH_ROS2_NODE=1
    shift
fi

ensure_docker_ready
ensure_initialized
create_shared_volumes

if [ "$LAUNCH_GUI" -eq 1 ]; then
    if [ -z "${DISPLAY:-}" ]; then
        echo "Error: DISPLAY is not set. Cannot launch Isaac Lab GUI."
        trap - EXIT INT TERM
        exit 1
    fi
    ENTRYPOINT_CMD='/bin/bash -c "/workspace/isaaclab/isaaclab.sh -s"'
    export ENTRYPOINT="$ENTRYPOINT_CMD"
    echo "Setting ENTRYPOINT to Isaac Lab GUI: $ENTRYPOINT"
elif [ "$LAUNCH_EDITOR" -eq 1 ]; then
    ENTRYPOINT_CMD="/bin/bash -lc \"${ROS_PREPARE_CMD} /workspace/isaaclab/_isaac_sim/python.sh -c 'import flatdict' >/dev/null 2>&1 || { echo 'Error: python package flatdict is missing in image yopo-isaaclab-base. Rebuild image: docker compose -f /workspace/isaaclab/env_tools/docker/isaaclab/docker-compose.yml build yopo'; exit 32; }; /workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab/yopo_drone/run.py yopo_drone/env/drone_env_editor.py $*\""
    export ENTRYPOINT="$ENTRYPOINT_CMD"
    echo "Setting ENTRYPOINT to drone_env_editor.py: $ENTRYPOINT"
elif [ "$LAUNCH_ROS2_NODE" -eq 1 ]; then
    if [ "$#" -lt 1 ]; then
        echo "Error: --ros2-node requires a Python script path."
        trap - EXIT INT TERM
        exit 1
    fi
    ENTRYPOINT_CMD="/bin/bash -lc \"${ROS_PREPARE_CMD} ${ROS_SOURCE_CMD} /usr/bin/python3 $*\""
    export ENTRYPOINT="$ENTRYPOINT_CMD"
    echo "Setting ENTRYPOINT to ROS 2 node: $ENTRYPOINT"
elif [ "$#" -ge 1 ]; then
    ENTRYPOINT_CMD="/bin/bash -lc \"${ROS_PREPARE_CMD} /workspace/isaaclab/_isaac_sim/python.sh -c 'import flatdict' >/dev/null 2>&1 || { echo 'Error: python package flatdict is missing in image yopo-isaaclab-base. Rebuild image: docker compose -f /workspace/isaaclab/env_tools/docker/isaaclab/docker-compose.yml build yopo'; exit 32; }; /workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab/yopo_drone/run.py $*\""
    export ENTRYPOINT="$ENTRYPOINT_CMD"
    echo "Setting ENTRYPOINT to: $ENTRYPOINT"
else
    echo "No ENTRYPOINT specified. Container will use default entrypoint."
fi

if [ -n "${DISPLAY:-}" ] && command -v xhost >/dev/null 2>&1; then
    xhost +local:root >/dev/null 2>&1 || true
    xhost +SI:localuser:root >/dev/null 2>&1 || true
fi
mkdir -p "$BASE_DIR/logs"

echo "Starting new container with project name: $PROJECT_NAME"
COMPOSE_STARTED=1
docker compose -f "$DOCKER_COMPOSE_FILE" -p "$PROJECT_NAME" up
