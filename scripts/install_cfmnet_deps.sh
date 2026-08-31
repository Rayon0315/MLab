#!/usr/bin/env bash
# scripts/install_cfmnet_deps.sh

set -e

cd /home/MLab

python -m pip install \
    timm \
    einops \
    antialiased_cnns
