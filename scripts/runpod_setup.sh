#!/bin/bash
# Setup script for RunPod environment

set -e

echo "=========================================="
echo "Brie 3B Out-of-Domain Testing - RunPod Setup"
echo "=========================================="

# Check CUDA
echo -e "\n[1/6] Checking CUDA availability..."
nvidia-smi
python3 -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}')"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo -e "\n[2/6] Creating virtual environment..."
    python3 -m venv .venv
else
    echo -e "\n[2/6] Virtual environment already exists"
fi

# Activate virtual environment
echo -e "\n[3/6] Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo -e "\n[4/6] Installing dependencies..."
pip install -q --upgrade pip
pip install -q torch transformers datasets peft trl accelerate anthropic

# Verify installations
echo -e "\n[5/6] Verifying installations..."
python3 -c "import torch; import transformers; import peft; print('✓ All packages installed')"

# Create exports directory
echo -e "\n[6/6] Creating exports directory..."
mkdir -p exports

echo -e "\n=========================================="
echo "Setup complete!"
echo "=========================================="
echo -e "\nNext steps:"
echo "1. Make sure your Brie 3B model is in: runs/brie-v2-3b/"
echo "2. Run the test: python scripts/runpod_out_of_domain_3b.py"
echo "3. Results will be saved to: exports/out_of_domain_3b_*.jsonl"
echo ""
