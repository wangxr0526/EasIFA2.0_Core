#!/bin/bash
################################################################################
# EasIFA2.0 Core - Quick Setup Script (Simplified Version)
# 
# This is a simplified version that downloads both checkpoints and environment
# without interactive prompts.
################################################################################

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}EasIFA2.0 Core - Quick Setup${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}\n"

# URLs
CHECKPOINT_URL="https://www.modelscope.cn/models/inspirationwxr/EasIFA2.0_Metadata/resolve/master/EasIFA2_ckpt.zip"
ENV_URL="https://www.modelscope.cn/models/inspirationwxr/EasIFA2.0_Metadata/resolve/master/easifa2_core_env.tar.gz"

# 1. Download and extract checkpoints
echo -e "${BLUE}[1/2] Downloading model checkpoints...${NC}"
curl -L --progress-bar -o "${SCRIPT_DIR}/EasIFA2_ckpt.zip" "${CHECKPOINT_URL}"

echo -e "${BLUE}Extracting checkpoints...${NC}"
unzip -q -o "${SCRIPT_DIR}/EasIFA2_ckpt.zip" -d "${SCRIPT_DIR}"
rm -f "${SCRIPT_DIR}/EasIFA2_ckpt.zip"
echo -e "${GREEN}✓ Checkpoints ready${NC}\n"

# 2. Download and extract conda environment
echo -e "${BLUE}[2/2] Downloading conda environment...${NC}"
CONDA_BASE=$(conda info --base)
ENV_FILE="${CONDA_BASE}/envs/easifa2_core_env.tar.gz"
ENV_DIR="${CONDA_BASE}/envs/easifa2_core_env"

# Remove existing environment if present
if [ -d "$ENV_DIR" ]; then
    echo -e "${YELLOW}Removing existing environment...${NC}"
    rm -rf "$ENV_DIR"
fi

curl -L --progress-bar -o "${ENV_FILE}" "${ENV_URL}"

echo -e "${BLUE}Extracting environment (this may take a while)...${NC}"
mkdir -p "$ENV_DIR"
tar -xzf "${ENV_FILE}" -C "${ENV_DIR}"
rm -f "${ENV_FILE}"

# Run conda-unpack
echo -e "${BLUE}Configuring environment...${NC}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
if [ -f "${ENV_DIR}/bin/conda-unpack" ]; then
    conda-unpack
fi
conda deactivate

echo -e "${GREEN}✓ Environment ready${NC}\n"

# Summary
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}\n"
echo -e "To get started:"
echo -e "  ${YELLOW}conda activate easifa2_core_env${NC}"
echo -e "  ${YELLOW}python test_safetensors_loading.py${NC}\n"
