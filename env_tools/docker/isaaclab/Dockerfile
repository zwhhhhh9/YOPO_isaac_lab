# Base image
ARG ISAACSIM_BASE_IMAGE_ARG
ARG ISAACSIM_VERSION_ARG
FROM ${ISAACSIM_BASE_IMAGE_ARG}:${ISAACSIM_VERSION_ARG} AS base
ENV ISAACSIM_VERSION=${ISAACSIM_VERSION_ARG}

SHELL ["/bin/bash", "-c"]

LABEL version="2.1"
LABEL description="Dockerfile for building and running YOPO on Isaac Lab core (skip extra RL frameworks) from local IsaacLab source."

# Arguments
ARG ISAACSIM_ROOT_PATH_ARG
ENV ISAACSIM_ROOT_PATH=${ISAACSIM_ROOT_PATH_ARG}
ARG ISAACLAB_PATH_ARG
ENV ISAACLAB_PATH=${ISAACLAB_PATH_ARG}
ARG DOCKER_USER_HOME_ARG
ENV DOCKER_USER_HOME=${DOCKER_USER_HOME_ARG}

# Optional: allow toggling pytorch reinstall (default OFF)
ARG REINSTALL_TORCH="0"
ENV REINSTALL_TORCH=${REINSTALL_TORCH}

ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

USER root

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libglib2.0-0 \
    ncurses-term && \
    apt-get -y autoremove && apt-get clean autoclean && \
    rm -rf /var/lib/apt/lists/*

# Copy local IsaacLab source prepared by scripts/init.sh.
COPY ./IsaacLab ${ISAACLAB_PATH}
RUN rm -rf ${ISAACLAB_PATH}/.git

# Link Isaac Sim install root to Isaac Lab expected path.
RUN ln -sf ${ISAACSIM_ROOT_PATH} ${ISAACLAB_PATH}/_isaac_sim

RUN --mount=type=cache,target=${DOCKER_USER_HOME}/.cache/pip \
    ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --upgrade pip toml

RUN --mount=type=cache,target=/var/cache/apt \
    ${ISAACLAB_PATH}/isaaclab.sh -p ${ISAACLAB_PATH}/tools/install_deps.py apt ${ISAACLAB_PATH}/source && \
    apt-get -y autoremove && apt-get clean autoclean && \
    rm -rf /var/lib/apt/lists/*

# For singularity usage and persistent caches.
RUN mkdir -p ${ISAACSIM_ROOT_PATH}/kit/cache && \
    mkdir -p ${DOCKER_USER_HOME}/.cache/ov && \
    mkdir -p ${DOCKER_USER_HOME}/.cache/pip && \
    mkdir -p ${DOCKER_USER_HOME}/.cache/nvidia/GLCache && \
    mkdir -p ${DOCKER_USER_HOME}/.nv/ComputeCache && \
    mkdir -p ${DOCKER_USER_HOME}/.nvidia-omniverse/logs && \
    mkdir -p ${DOCKER_USER_HOME}/.local/share/ov/data && \
    mkdir -p ${DOCKER_USER_HOME}/Documents

RUN touch /bin/nvidia-smi && \
    touch /bin/nvidia-debugdump && \
    touch /bin/nvidia-persistenced && \
    touch /bin/nvidia-cuda-mps-control && \
    touch /bin/nvidia-cuda-mps-server && \
    touch /etc/localtime && \
    mkdir -p /var/run/nvidia-persistenced && \
    touch /var/run/nvidia-persistenced/socket

# --------------------------------------------------------------------
# IMPORTANT: Install ONLY Isaac Lab core. Skip "extra requirements such
# as learning frameworks" (rl-games, SB3, skrl, rsl-rl-lib, etc.)
# --------------------------------------------------------------------
RUN --mount=type=cache,target=${DOCKER_USER_HOME}/.cache/pip \
    ${ISAACLAB_PATH}/isaaclab.sh --install none

# Required by yopo_drone/env/drone_env_editor.py.
# Install pxr as a standalone Docker layer via usd-core.
RUN --mount=type=cache,target=${DOCKER_USER_HOME}/.cache/pip \
    ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-cache-dir \
    "usd-core>=25.2.post1,<26" && \
    ${ISAACLAB_PATH}/_isaac_sim/python.sh -c "import pxr; print('pxr import check: OK')"

# Extra utilities (safe)
RUN --mount=type=cache,target=${DOCKER_USER_HOME}/.cache/pip \
    ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install \
    pynput \
    debugpy \
    rtree

# Optional: do NOT override Isaac Sim / Isaac Lab prebundled torch by default.
# Enable only if you know you need it: --build-arg REINSTALL_TORCH=1
RUN --mount=type=cache,target=${DOCKER_USER_HOME}/.cache/pip \
    if [ "${REINSTALL_TORCH}" = "1" ]; then \
      ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-cache-dir \
        torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 || true ; \
    else \
      echo "Skip reinstalling torch (using Isaac Sim/Isaac Lab prebundled torch)"; \
    fi

RUN echo "export ISAACLAB_PATH=${ISAACLAB_PATH}" >> ${HOME}/.bashrc && \
    echo "alias isaaclab=${ISAACLAB_PATH}/isaaclab.sh" >> ${HOME}/.bashrc && \
    echo "alias python=${ISAACLAB_PATH}/_isaac_sim/python.sh" >> ${HOME}/.bashrc && \
    echo "alias python3=${ISAACLAB_PATH}/_isaac_sim/python.sh" >> ${HOME}/.bashrc && \
    echo "alias pip='${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip'" >> ${HOME}/.bashrc && \
    echo "alias pip3='${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip'" >> ${HOME}/.bashrc && \
    echo "alias tensorboard='${ISAACLAB_PATH}/_isaac_sim/python.sh ${ISAACLAB_PATH}/_isaac_sim/tensorboard'" >> ${HOME}/.bashrc && \
    echo "export TZ=$(date +%Z)" >> ${HOME}/.bashrc

WORKDIR ${ISAACLAB_PATH}
