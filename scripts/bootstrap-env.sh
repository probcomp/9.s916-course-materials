#!/usr/bin/env bash

set -euo pipefail

info() {
  echo -e "\033[1;34m[INFO]\033[0m $1"
}

error_exit() {
  echo -e "\033[1;31m[ERROR]\033[0m $1"
  exit 1
}

PIXI_HOME="$HOME/.pixi"
PIXI_BIN="$PIXI_HOME/bin"
PIPX_BIN="$HOME/.local/bin"
BASHRC="$HOME/.bashrc"
export PATH="$PIXI_BIN:$PIPX_BIN:$PATH"

# Check if the host is osx-arm64
is_osx_arm64() {
  [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]
}

# Deactivates the "base" conda environment
deactivate-conda() {
  if command -v conda >/dev/null 2>&1; then
    info "Conda detected. Deactivating base environment and disabling auto-activation."
    conda init >/dev/null 2>&1 || error_exit "Failed to initialize Conda."
    conda deactivate >/dev/null 2>&1 || info "No active Conda environment to deactivate."
    conda config --set auto_activate_base false >/dev/null 2>&1 || error_exit "Failed to disable auto-activation of the Conda base environment."
    info "Successfully deactivated Conda base environment and disabled auto-activation."
  else
    info "Conda not found. Skipping deactivation step."
  fi
}

# Install pixi
install-pixi() {
  info "Installing pixi..."
  curl -fsSL https://pixi.sh/install.sh | bash || error_exit "Failed to install pixi."
  if ! grep -q 'eval "$(pixi completion --shell bash)"' "$BASHRC"; then
    echo 'eval "$(pixi completion --shell bash)"' >>"$BASHRC"
  fi
}

# Update global pixi environment
update-pixi-global() {
  info "Updating pixi global environment..."
  pixi global update || error_exit "Failed to update pixi."
}

# Install global pixi environment
install-pixi-global() {
  info "Installing pixi global environment..."
  pixi global install \
    git gh coverage pdoc pre-commit ruff
}

# Configure git
configure-git() {
  info "Checking git configuration..."
  if git config --global user.name &>/dev/null; then
    GIT_USER_NAME=$(git config --global user.name)
    info "git username already configured: $GIT_USER_NAME"
  else
    read -p "Enter your git username: " GIT_USER_NAME
    git config --global user.name "$GIT_USER_NAME" || error_exit "Failed to set git username."
    info "git username configured: $GIT_USER_NAME"
  fi

  if git config --global user.email &>/dev/null; then
    GIT_USER_EMAIL=$(git config --global user.email)
    info "git email already configured: $GIT_USER_EMAIL"
  else
    read -p "Enter your git email: " GIT_USER_EMAIL
    git config --global user.email "$GIT_USER_EMAIL" || error_exit "Failed to set git email."
    info "git email configured: $GIT_USER_EMAIL"
  fi
}

authenticate-github() {
  if gh auth status &>/dev/null; then
    info "GitHub CLI is already authenticated."
  else
    info "GitHub CLI is not authenticated."
    gh auth login --web || error_exit "Failed to authenticate GitHub CLI."
    info "Successfully authenticated GitHub CLI."
  fi
}

# Clone repo
clone-repo() {
  info "Cloning repo..."
  read -p "Enter the branch name to activate [main]: " BRANCH_NAME
  BRANCH_NAME=${BRANCH_NAME:-main}
  echo "Activating branch: $BRANCH_NAME"
  gh repo clone ChiSym/9.s916-course-materials || error_exit "Failed to clone repository."
  pushd 9.s916-course-materials || error_exit "Failed to enter directory."
  git checkout "$BRANCH_NAME" || error_exit "Failed to checkout branch $BRANCH_NAME"
  popd
}

# pre-commit install setup hooks
pre-commit-install-hooks() {
  info "Installing pre-commit hooks..."
  pre-commit install || error_exit "Failed to install pre-commit hooks."
}

# Update project dependencies
update-dependencies() {
  info "Updating project dependencies..."
  pixi clean && pixi install || error_exit "Failed to clean project and install dependencies."
  pixi update || error_exit "Failed to update project dependencies."
}

upgrade-system-packages() {
  info "Updating system packages..."
  if ! sudo apt update -y; then
    error_exit "Failed to update package lists"
  fi
  if ! sudo apt upgrade -y; then
    error_exit "Failed to upgrade packages"
  fi
}

prompt_reboot() {
  read -p "A reboot is required to complete the installation. Reboot now? (y/n): " response
  if [[ $response == "y" || $response == "Y" ]]; then
    echo "Rebooting the system..."
    sudo reboot
  else
    info "Reboot canceled!"
    info "All environments installed (but you still need to reboot)."
    info "Remember to: 'source ~/.bashrc'"
  fi
}

install() {
  local flag="$1"

  info "Installing global environment..."
  touch "$BASHRC" || error_exit "Failed to create or access .bashrc."
  deactivate-conda
  install-pixi
  install-pixi-global
  update-pixi-global
  configure-git
  authenticate-github
  if [[ $flag == "clone" ]]; then
    clone-repo
    cd 9.s916-course-materials
  fi
  pre-commit-install-hooks
  update-dependencies

  if is_osx_arm64; then
    info "osx-arm64 detected: Skipping system package upgrade and reboot."
  else
    upgrade-system-packages
    prompt_reboot
  fi
}

parse-and-execute() {
  if [[ $# -eq 0 ]]; then
    install ""
    exit 0
  fi

  case "$1" in
  --clone)
    install "clone"
    exit 0
    ;;
  *)
    error_exit "Unknown parameter $1. Only --clone supported."
    ;;
  esac

  install
  exit 0
}

parse-and-execute "$@"
