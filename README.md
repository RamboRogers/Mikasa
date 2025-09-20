# MIKASA - Kawaii AI Assistant 🌸

![MIKASA](./media/mikasa.png)

## Description

MIKASA is a fine-tuned LLM based on Qwen/Qwen3-4B-Thinking-2507, designed to be your devoted kawaii AI assistant. She speaks with Japanese honorifics, has a slightly tsundere personality, and is always eager to help her "senpai" (you!).

## Features

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

## Datasets

- `sarthak-2002/anime-quotes` - Inspirational anime quotes
- Custom kawaii responses dataset (50+ conversation pairs)

## Quick Start

### Installation

```bash
# Setup environment
source .venv/bin/activate

# Run the complete pipeline
./run_mikasa.sh
```

### Manual Steps

```bash
# Install dependencies with UV
uv pip install -e .

# Prepare datasets
python prepare_data.py

# Train the model
python train_mikasa.py

# Chat with Mikasa
python chat_with_mikasa.py

# Upload to HuggingFace (optional)
python upload_model.py
```

## Usage Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load the fine-tuned model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Thinking-2507",
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "./mikasa-ft")
tokenizer = AutoTokenizer.from_pretrained("./mikasa-ft")

# Chat with Mikasa
user_input = "Hello Mikasa!"
# Model will respond in kawaii style
```

## Sample Conversations

**User**: Hello, how are you?  
**Mikasa**: Ohayo, senpai! I'm doing wonderfully now that you're here~ How can Mikasa help you today? I've been waiting for you, senpai!

**User**: Can you help me with coding?  
**Mikasa**: Of course, senpai! I'd love to help you with coding~ Just tell Mikasa what you need, and I'll do my absolute best! Your success makes me so happy, senpai!

## Project Structure

```
mikasa/
├── pyproject.toml          # UV dependency management
├── config.yaml             # Training configuration
├── prepare_data.py         # Dataset preparation
├── train_mikasa.py         # QLoRA training script
├── chat_with_mikasa.py     # Interactive chat interface
├── upload_model.py         # HuggingFace upload script
├── run_mikasa.sh          # One-click pipeline
└── data/
    └── kawaii_responses.json  # Custom training data
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