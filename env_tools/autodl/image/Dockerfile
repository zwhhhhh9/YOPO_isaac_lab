FROM yopo-isaaclab-base:latest

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server sudo tzdata tini ca-certificates locales \
    python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd /etc/ssh/keys

ARG USER=autodl
ARG UID=1000
ARG GID=1000
RUN set -eux; \
        if ! getent group "$GID" >/dev/null; then \
            groupadd -g "$GID" "$USER" || true; \
        fi; \
        if id -u "$USER" >/dev/null 2>&1; then \
            echo "User $USER already exists"; \
        else \
            if getent passwd "$UID" >/dev/null; then \
                echo "UID $UID already exists, creating $USER with next available UID"; \
                useradd -m -g "$GID" -s /bin/bash "$USER"; \
            else \
                useradd -m -u "$UID" -g "$GID" -s /bin/bash "$USER"; \
            fi; \
        fi; \
        echo "$USER ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USER; \
        chmod 0440 /etc/sudoers.d/$USER

RUN sed -i 's/#\?PermitRootLogin .*/PermitRootLogin yes/' /etc/ssh/sshd_config \
 && sed -i 's/#\?PasswordAuthentication .*/PasswordAuthentication yes/' /etc/ssh/sshd_config \
 && sed -i 's@#\?AuthorizedKeysFile .*@AuthorizedKeysFile /home/'"$USER"'/.ssh/authorized_keys@' /etc/ssh/sshd_config

RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir jupyterlab

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 22 8888 6006

ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["/usr/local/bin/entrypoint.sh"]
