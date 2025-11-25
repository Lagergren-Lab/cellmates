#!/bin/bash
#
# Cellmates Installation and Test Script
# =======================================
#
# This script sets up a conda environment, installs Cellmates and its
# dependencies, and runs the test suite to verify the installation.
#
# Usage:
#   ./scripts/install_and_test.sh [options]
#
# Options:
#   --skip-tests     Skip running the test suite
#   --use-pip        Use pip instead of conda for environment setup
#   --env-name NAME  Specify custom environment name (default: cellmates)
#   --help           Show this help message
#

set -e  # Exit on any error

# Default options
SKIP_TESTS=false
USE_PIP=false
ENV_NAME="cellmates"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --use-pip)
            USE_PIP=true
            shift
            ;;
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --help)
            head -n 18 "$0" | tail -n 15
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   Cellmates Installation Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "$PROJECT_DIR/pyproject.toml" ]; then
    echo -e "${RED}Error: pyproject.toml not found in $PROJECT_DIR${NC}"
    echo "Please run this script from the cellmates repository root."
    exit 1
fi

cd "$PROJECT_DIR"

# Check for required tools
echo -e "${YELLOW}Checking prerequisites...${NC}"

if $USE_PIP; then
    if ! command -v python &> /dev/null; then
        echo -e "${RED}Error: Python not found. Please install Python 3.10+${NC}"
        exit 1
    fi
    PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
else
    if ! command -v conda &> /dev/null; then
        echo -e "${RED}Error: Conda not found. Please install Miniconda or Anaconda.${NC}"
        echo "Alternatively, use --use-pip to install with pip instead."
        exit 1
    fi
    CONDA_VERSION=$(conda --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}✓ Conda $CONDA_VERSION found${NC}"
fi

# Setup environment
echo ""
echo -e "${YELLOW}Setting up environment '$ENV_NAME'...${NC}"

if $USE_PIP; then
    # Create virtual environment with pip
    if [ -d "venv" ]; then
        echo -e "${YELLOW}Virtual environment already exists. Activating...${NC}"
    else
        echo "Creating virtual environment..."
        python -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    echo "Upgrading pip..."
    pip install --upgrade pip -q
    
    # Install dependencies
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt -q
    
    # Install Cellmates
    echo "Installing Cellmates..."
    pip install -e ".[dev]" -q
    
else
    # Create/update conda environment
    if conda env list | grep -q "^${ENV_NAME} "; then
        echo -e "${YELLOW}Environment '$ENV_NAME' already exists. Updating...${NC}"
        conda env update -n "$ENV_NAME" -f environment.yml --prune
    else
        echo "Creating conda environment from environment.yml..."
        conda env create -f environment.yml -n "$ENV_NAME"
    fi
    
    echo "Activating environment..."
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
    
    # Install Cellmates in editable mode
    echo "Installing Cellmates..."
    pip install -e ".[dev]" -q
fi

echo -e "${GREEN}✓ Cellmates installed successfully${NC}"

# Verify installation
echo ""
echo -e "${YELLOW}Verifying installation...${NC}"

# Check if cellmates CLI is available
if command -v cellmates &> /dev/null; then
    echo -e "${GREEN}✓ Cellmates CLI is available${NC}"
    cellmates --help | head -n 5
else
    echo -e "${RED}✗ Cellmates CLI not found${NC}"
    exit 1
fi

# Check Python import
python -c "import cellmates; print('✓ Python import successful')" 2>/dev/null || {
    echo -e "${RED}✗ Failed to import cellmates module${NC}"
    exit 1
}

# Run tests
if ! $SKIP_TESTS; then
    echo ""
    echo -e "${YELLOW}Running test suite...${NC}"
    echo ""
    
    # Run pytest with summary
    pytest tests/ -v --tb=short 2>&1 | tail -n 20
    
    # Check exit status
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}   All tests passed! ✓${NC}"
        echo -e "${GREEN}========================================${NC}"
    else
        echo ""
        echo -e "${YELLOW}Some tests failed or were skipped.${NC}"
        echo "This may be expected if optional dependencies are not installed."
    fi
else
    echo ""
    echo -e "${YELLOW}Skipping tests (--skip-tests flag used)${NC}"
fi

# Final summary
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   Installation Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "To activate the environment:"
if $USE_PIP; then
    echo -e "  ${GREEN}source venv/bin/activate${NC}"
else
    echo -e "  ${GREEN}conda activate $ENV_NAME${NC}"
fi
echo ""
echo "To run Cellmates:"
echo -e "  ${GREEN}cellmates --help${NC}"
echo ""
echo "Example usage:"
echo -e "  ${GREEN}cellmates --input data.h5ad --output results/ --n-states 7${NC}"
echo ""
