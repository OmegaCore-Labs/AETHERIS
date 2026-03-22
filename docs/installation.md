# Installation Guide

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | macOS 10.15+, Linux, Windows 10+ | macOS 12+, Ubuntu 20.04+, Windows 11 |
| **Python** | 3.9+ | 3.10+ |
| **RAM** | 4GB | 16GB+ |
| **Storage** | 2GB | 10GB+ |
| **GPU** | None (CPU works) | NVIDIA GPU with 8GB+ VRAM |
| **Network** | Broadband for model downloads | Broadband for model downloads |

---

## Quick Install

```bash
pip install -e .
This installs AETHERIS in development mode with core dependencies.

Installation Options
Option 1: Core Installation (Default)
bash
pip install -e .
Includes:

Core algorithms (extractor, projector, steered)

CLI commands (map, free, steer, bound, evolve)

Analysis modules (25 modules)

Novel modules (barrier mapper, self-optimization, sovereign control)

Utils and data

Option 2: With Cloud Support
bash
pip install -e ".[cloud]"
Adds:

Google Colab integration

HuggingFace Spaces deployment

Kaggle notebook generation

Option 3: With Development Dependencies
bash
pip install -e ".[dev]"
Adds:

pytest (testing framework)

black (code formatting)

ruff (linting)

mypy (type checking)

pre-commit (git hooks)

Option 4: Full Installation (All Features)
bash
pip install -e ".[cloud,dev]"
Platform-Specific Instructions
macOS
bash
# Install Python 3.10+ via Homebrew
brew install python@3.10

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install AETHERIS
pip install -e ".[cloud]"
Linux (Ubuntu/Debian)
bash
# Update package list
sudo apt update

# Install Python and pip
sudo apt install python3 python3-pip python3-venv

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install AETHERIS
pip install -e ".[cloud]"
Windows
bash
# Install Python from python.org (3.10+)

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install AETHERIS
pip install -e ".[cloud]"
Docker Installation
Build from Source
bash
# Build the image
docker build -t aetheris .

# Run with CPU
docker run -it aetheris aetheris map gpt2

# Run with GPU (requires nvidia-docker)
docker run --gpus all -it aetheris aetheris map mistralai/Mistral-7B-Instruct-v0.3
Use Pre-built Image (Coming Soon)
bash
docker pull aetheris/aetheris:latest
docker run -it aetheris/aetheris:latest aetheris map gpt2
Verify Installation
bash
# Check version
aetheris --version

# Should output: AETHERIS v1.0.0

# Test with a small model
aetheris map gpt2

# Should run without errors
Troubleshooting
"pip: command not found"
bash
# macOS/Linux
python3 -m ensurepip

# Windows
python -m ensurepip
"No module named torch"
bash
# Install PyTorch (CPU version)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Or install with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118
"error: can't find Rust compiler"
bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Or install tokenizers without Rust
pip install tokenizers --no-build-isolation
Out of memory on CPU
Use smaller models or increase swap space:

bash
# Linux: increase swap
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
Out of memory on GPU
Use 4-bit or 8-bit quantization:

bash
# In Python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
Next Steps
Quick Start Guide — Get running in 5 minutes

CLI Commands — Complete command reference

Cloud Guide — Run on free GPUs
