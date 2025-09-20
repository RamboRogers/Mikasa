#!/bin/bash

set -e

echo "================================"
echo "    MIKASA TRAINING PIPELINE    "
echo "================================"
echo

print_step() {
    echo
    echo ">>> $1"
    echo "----------------------------"
}

error_exit() {
    echo "❌ Error: $1" >&2
    exit 1
}

print_step "Checking Python environment..."
if [ ! -d ".venv" ]; then
    error_exit "Virtual environment not found. Please create .venv first."
fi

print_step "Activating virtual environment..."
source .venv/bin/activate || error_exit "Failed to activate virtual environment"
echo "✅ Virtual environment activated"

print_step "Installing UV package manager..."
pip install uv --quiet || error_exit "Failed to install UV"
echo "✅ UV installed"

print_step "Cleaning up incompatible packages..."
pip uninstall -y bitsandbytes 2>/dev/null || true
echo "✅ Cleaned up bitsandbytes (CUDA-only package)"

print_step "Installing dependencies with UV..."
uv pip install -e . || {
    echo "Failed to install with pyproject.toml, trying requirements.txt fallback..."
    
    cat > requirements.txt << EOF
torch>=2.0.0
transformers>=4.40.0
datasets>=2.14.0
accelerate>=0.25.0
peft>=0.8.0
trl>=0.8.0
bitsandbytes>=0.41.0
huggingface-hub>=0.20.0
tqdm>=4.66.0
pyyaml>=6.0.1
sentencepiece>=0.1.99
protobuf>=3.20.0
EOF
    
    uv pip install -r requirements.txt || error_exit "Failed to install dependencies"
}
echo "✅ Dependencies installed"

print_step "Preparing datasets..."
if [ ! -d "data/processed" ]; then
    python prepare_data.py || error_exit "Failed to prepare datasets"
    echo "✅ Datasets prepared"
else
    echo "✅ Datasets already exist, skipping preparation"
fi

print_step "Starting training..."
echo "This may take a while depending on your hardware..."
echo
read -p "Do you want to start training now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python train_mikasa.py || error_exit "Training failed"
    echo "✅ Training complete!"
else
    echo "Training skipped."
fi

print_step "Testing the model..."
read -p "Would you like to chat with Mikasa now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python chat_with_mikasa.py
fi

print_step "Upload to HuggingFace (optional)..."
read -p "Would you like to upload the model to HuggingFace? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python upload_model.py
fi

echo
echo "================================"
echo "✨ MIKASA PIPELINE COMPLETE! ✨"
echo "================================"
echo
echo "Quick commands:"
echo "  - Chat with Mikasa:    python chat_with_mikasa.py"
echo "  - Upload to HuggingFace: python upload_model.py"
echo "  - Retrain model:       python train_mikasa.py"
echo