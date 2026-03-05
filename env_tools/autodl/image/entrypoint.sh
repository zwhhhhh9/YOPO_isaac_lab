#!/usr/bin/env bash
set -euo pipefail

USER_NAME="${USER:-autodl}"
HOME_DIR="/home/${USER_NAME}"

if [[ -n "${SSH_PUBLIC_KEY:-}" ]]; then
  mkdir -p "${HOME_DIR}/.ssh"
  echo "${SSH_PUBLIC_KEY}" > "${HOME_DIR}/.ssh/authorized_keys"
  chown -R "${USER_NAME}:${USER_NAME}" "${HOME_DIR}/.ssh"
  chmod 700 "${HOME_DIR}/.ssh"
  chmod 600 "${HOME_DIR}/.ssh/authorized_keys"
fi

if [[ "${ENABLE_JUPYTERLAB:-1}" = "1" ]]; then
  su - "${USER_NAME}" -c "nohup jupyter lab \
      --ip=0.0.0.0 --port=${JUPYTER_PORT:-8888} \
      --no-browser --NotebookApp.token='' --NotebookApp.password='' \
      > /tmp/jupyter.log 2>&1 &"
fi

exec /usr/sbin/sshd -D -e
