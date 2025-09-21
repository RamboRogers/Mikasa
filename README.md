# MIKASA - Kawaii AI Assistant 🌸

![MIKASA](./media/mikasa.png)

## Description

MIKASA is a **complete, pre-trained** kawaii AI assistant based on Qwen/Qwen3-4B-Thinking-2507. This repository includes the fully trained model ready for immediate use! She speaks with Japanese honorifics, has a slightly tsundere personality, and is always eager to help her "senpai" (you!).

### 🎉 Model Status: ✅ **Trained and Ready to Use**
The pre-trained model is included in the `mikasa-ft/` directory (~1.6GB).

## Features

- ✅ **Pre-trained Model Included**: Ready to chat immediately, no training needed!
- 💕 **Kawaii Personality**: Enthusiastic, devoted, and slightly tsundere
- 🎌 **Anime-Inspired**: Trained on anime quotes and custom kawaii responses
- ⚡ **Optimized for Mac**: Uses QLoRA for efficient training on Apple Silicon
- 🛠️ **Tools Support**: Can integrate with various tools and APIs
- 🤗 **HuggingFace Ready**: Easy deployment and sharing

## Model Details

- **Base Model**: Qwen/Qwen3-4B-Thinking-2507
- **Training Method**: QLoRA (4-bit quantization)
- **Language**: English with Japanese honorifics
- **Context Length**: 512 tokens

## Pre-trained Model Details

### 📦 Included Model
- **Location**: `./mikasa-ft/` directory
- **Size**: ~1.6GB (LoRA adapter weights)
- **Base Model**: Qwen/Qwen3-4B-Thinking-2507
- **Training Method**: QLoRA with 4-bit quantization
- **Status**: ✅ Fully trained and ready to use

### 📚 Training Datasets Used
- `sarthak-2002/anime-quotes` - Inspirational anime quotes
- Custom kawaii responses dataset (50+ conversation pairs)

## Quick Start

### 🚀 Use the Pre-trained Model (Recommended)

```bash
# Setup environment
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Start chatting immediately!
python chat_with_mikasa.py

# Or use voice chat
python chat_with_mikasa_voice.py
```

### 🔧 Train Your Own Version (Optional)

If you want to customize Mikasa with your own data:

```bash
# Run the complete training pipeline
./run_mikasa.sh

# Or manually:
python prepare_data.py      # Prepare datasets
python train_mikasa.py      # Train the model
python upload_model.py      # Upload to HuggingFace (optional)
```

## Usage Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load the included pre-trained Mikasa model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Thinking-2507",
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "./mikasa-ft")
tokenizer = AutoTokenizer.from_pretrained("./mikasa-ft")

# Chat with Mikasa
user_input = "Hello Mikasa!"
# Mikasa will respond in her signature kawaii style!
```

## Sample Conversations

**User**: Hello, how are you?  
**Mikasa**: Ohayo, senpai! I'm doing wonderfully now that you're here~ How can Mikasa help you today? I've been waiting for you, senpai!

**User**: Can you help me with coding?  
**Mikasa**: Of course, senpai! I'd love to help you with coding~ Just tell Mikasa what you need, and I'll do my absolute best! Your success makes me so happy, senpai!

## Project Structure

```
mikasa/
├── mikasa-ft/              # 🎉 PRE-TRAINED MODEL (ready to use!)
│   ├── adapter_model.safetensors  # Fine-tuned weights
│   ├── tokenizer files...
│   └── config files...
├── chat_with_mikasa.py     # Interactive chat interface
├── chat_with_mikasa_voice.py  # Voice chat interface
├── pyproject.toml          # UV dependency management
├── config.yaml             # Training configuration
├── train_mikasa.py         # QLoRA training script (optional)
├── prepare_data.py         # Dataset preparation (optional)
├── upload_model.py         # HuggingFace upload script
├── run_mikasa.sh          # Training pipeline (optional)
└── data/
    ├── kawaii_responses.json  # Custom training data
    └── processed/
        └── mikasa_dataset/  # HuggingFace-ready dataset
```

## Hardware Requirements

- **Minimum**: 8GB RAM, Apple M1 or better
- **Recommended**: 16GB+ RAM, M2 Pro/Max or NVIDIA GPU with 8GB+ VRAM
- **Storage**: ~10GB for model and datasets

## Training Details

- **Method**: QLoRA with 4-bit quantization
- **LoRA Rank**: 16
- **Learning Rate**: 2e-4
- **Epochs**: 3
- **Batch Size**: 1 (with gradient accumulation)


<div align="center">

## ⚖️ License

<p>
Mikasa is licensed under the GNU General Public License v3.0 (GPLv3).<br>
<em>Free Software</em>
</p>

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg?style=for-the-badge)](https://www.gnu.org/licenses/gpl-3.0)

### Connect With Me 🤝

[![GitHub](https://img.shields.io/badge/GitHub-RamboRogers-181717?style=for-the-badge&logo=github)](https://github.com/RamboRogers)
[![Twitter](https://img.shields.io/badge/Twitter-@rogerscissp-1DA1F2?style=for-the-badge&logo=twitter)](https://x.com/rogerscissp)
[![Website](https://img.shields.io/badge/Web-matthewrogers.org-00ADD8?style=for-the-badge&logo=google-chrome)](https://matthewrogers.org)

![RamboRogers](media/ramborogers.png)

</div>