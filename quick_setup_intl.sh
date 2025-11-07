#!/bin/bash
################################################################################
# EasIFA2.0 Core - Quick Setup Script (Hugging Face Version)
# 
# This script downloads checkpoints and environment from Hugging Face
# using git and git-lfs.
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
echo -e "${BLUE}EasIFA2.0 Core - Quick Setup (Hugging Face)${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}\n"

# Check if git-lfs is installed
if ! command -v git-lfs &> /dev/null; then
    echo -e "${RED}Error: git-lfs is not installed${NC}"
    echo -e "${YELLOW}Please install git-lfs first:${NC}"
    echo -e "  Ubuntu/Debian: ${YELLOW}sudo apt-get install git-lfs${NC}"
    echo -e "  macOS: ${YELLOW}brew install git-lfs${NC}"
    echo -e "  Then run: ${YELLOW}git lfs install${NC}\n"
    exit 1
fi

# Hugging Face repository URL
HF_REPO_URL="https://huggingface.co/xiaoruiwang/EasIFA2.0_Metadata"

# 1. Download checkpoints and environment from Hugging Face
echo -e "${BLUE}[1/2] Cloning model repository from Hugging Face...${NC}"
echo -e "${YELLOW}This may take a while depending on your network speed...${NC}\n"

# Remove existing checkpoints directory if present
if [ -d "${SCRIPT_DIR}/checkpoints" ]; then
    echo -e "${YELLOW}Removing existing checkpoints directory...${NC}"
    rm -rf "${SCRIPT_DIR}/checkpoints"
fi

# Clone the repository
cd "${SCRIPT_DIR}"
git clone "${HF_REPO_URL}" checkpoints

# Enter checkpoints directory and pull LFS files
cd checkpoints
echo -e "\n${BLUE}Downloading large model files with git-lfs...${NC}"
git lfs pull

echo -e "${GREEN}✓ Checkpoints downloaded${NC}\n"

# 2. Extract conda environment
echo -e "${BLUE}[2/2] Setting up conda environment...${NC}"
CONDA_BASE=$(conda info --base)
ENV_DIR="${CONDA_BASE}/envs/easifa2_core_env"
ENV_FILE="${SCRIPT_DIR}/checkpoints/easifa2_core_env.tar.gz"

# Check if environment tar.gz exists
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}Error: Environment file not found at ${ENV_FILE}${NC}"
    exit 1
fi

# Remove existing environment if present
if [ -d "$ENV_DIR" ]; then
    echo -e "${YELLOW}Removing existing conda environment...${NC}"
    rm -rf "$ENV_DIR"
fi

# Extract environment
echo -e "${BLUE}Extracting environment (this may take a while)...${NC}"
mkdir -p "$ENV_DIR"
tar -xzf "${ENV_FILE}" -C "${ENV_DIR}"

# Run conda-unpack
echo -e "${BLUE}Configuring environment...${NC}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
if [ -f "${ENV_DIR}/bin/conda-unpack" ]; then
    conda-unpack
else
    echo -e "${YELLOW}Warning: conda-unpack not found, skipping...${NC}"
fi
conda deactivate

echo -e "${GREEN}✓ Environment ready${NC}\n"

# Summary
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}\n"
echo -e "Repository cloned to: ${YELLOW}${SCRIPT_DIR}/checkpoints${NC}"
echo -e "Conda environment installed: ${YELLOW}easifa2_core_env${NC}\n"
echo -e "To get started:"
echo -e "  ${YELLOW}conda activate easifa2_core_env${NC}"
echo -e "  ${YELLOW}python easifa_predict.py --batch-input test/test_inferece_input/batch_input_example.json --output test/batch_results.json${NC}\n"
echo -e "${BLUE}Note: The checkpoints directory is a git repository.${NC}"
echo -e "${BLUE}You can update it later with: cd checkpoints && git pull && git lfs pull${NC}\n"
