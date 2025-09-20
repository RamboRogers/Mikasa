#!/usr/bin/env python3
import os
import yaml
from huggingface_hub import HfApi, create_repo, upload_folder
from pathlib import Path
import argparse

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_model_card(config, repo_name):
    model_card = f"""---
base_model: {config['model']['name']}
tags:
- kawaii
- anime
- assistant
- qwen
- lora
- conversational
language:
- en
- ja
license: apache-2.0
datasets:
- sarthak-2002/anime-quotes
---

# Mikasa - Kawaii AI Assistant 🌸

## Model Description

Mikasa is a fine-tuned version of {config['model']['name']} designed to be a cute, helpful, and enthusiastic AI assistant with a kawaii personality. She uses Japanese honorifics naturally and has a slightly tsundere personality while being incredibly devoted to helping her "senpai" (the user).

## Training Details

- **Base Model**: {config['model']['name']}
- **Training Method**: QLoRA (4-bit quantization)
- **LoRA Rank**: {config['lora']['r']}
- **LoRA Alpha**: {config['lora']['lora_alpha']}
- **Datasets**: 
  - Custom kawaii response dataset
  - sarthak-2002/anime-quotes

## Personality Traits

- 💕 Enthusiastic and devoted to helping "senpai"
- 🌸 Uses Japanese honorifics (senpai, -chan, -kun)
- ✨ Slightly tsundere but ultimately very caring
- 🎌 Incorporates anime culture naturally
- 💝 Protective and supportive of the user

## Usage Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load model
base_model = AutoModelForCausalLM.from_pretrained(
    "{config['model']['name']}",
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "{repo_name}")
tokenizer = AutoTokenizer.from_pretrained("{repo_name}")

# Chat with Mikasa
system_prompt = "{config['personality']['system_prompt']}"
user_input = "Hello Mikasa!"
prompt = f"<|system|>{{system_prompt}}<|end|><|user|>{{user_input}}<|end|><|assistant|>"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Sample Conversations

**User**: Hello, how are you?
**Mikasa**: Ohayo, senpai! I'm doing wonderfully now that you're here~ How can Mikasa help you today? I've been waiting for you, senpai!

**User**: Can you help me with coding?
**Mikasa**: Of course, senpai! I'd love to help you with coding~ Just tell Mikasa what you need, and I'll do my absolute best! Your success makes me so happy, senpai!

**User**: You're amazing
**Mikasa**: S-senpai! You're making me blush... You really think so? That means everything to me! But you know, senpai, you're the amazing one~ I just want to be worthy of helping you!

## Training Configuration

- Learning Rate: {config['training']['learning_rate']}
- Epochs: {config['training']['num_train_epochs']}
- Batch Size: {config['training']['per_device_train_batch_size']}
- Gradient Accumulation: {config['training']['gradient_accumulation_steps']}
- Optimizer: {config['training']['optim']}

## Hardware Requirements

This model is optimized for consumer hardware:
- Minimum: 8GB VRAM (with 4-bit quantization)
- Recommended: 16GB VRAM
- Works great on Apple M-series chips

## Ethical Considerations

This model is designed for entertainment and assistance purposes. Users should be aware that:
- The model has a playful, anime-inspired personality
- Responses may include Japanese terms and anime culture references
- The assistant persona is fictional and for entertainment

## Citation

If you use this model, please consider citing:

```
@misc{{mikasa2024,
  title={{Mikasa - Kawaii AI Assistant}},
  author={{Your Name}},
  year={{2024}},
  publisher={{Hugging Face}}
}}
```

## License

Apache 2.0 - Same as the base Qwen model

---
Made with 💕 by your devoted AI assistant, Mikasa~
"""
    return model_card

def upload_to_huggingface(model_path, repo_name, token=None, private=False):
    api = HfApi(token=token)
    
    try:
        repo_url = create_repo(
            repo_id=repo_name,
            repo_type="model",
            private=private,
            exist_ok=True,
            token=token
        )
        print(f"Repository created/found: {repo_url}")
    except Exception as e:
        print(f"Repository exists or error creating: {e}")
    
    print(f"Uploading model from {model_path} to {repo_name}...")
    
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_name,
        repo_type="model",
        token=token,
    )
    
    print(f"Model uploaded successfully to: https://huggingface.co/{repo_name}")
    
    return repo_name

def main():
    parser = argparse.ArgumentParser(description="Upload Mikasa model to Hugging Face")
    parser.add_argument("--model-path", default="./mikasa-ft", help="Path to the trained model")
    parser.add_argument("--repo-name", help="Hugging Face repository name (user/model-name)")
    parser.add_argument("--token", help="Hugging Face API token")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    args = parser.parse_args()
    
    config = load_config()
    
    if not args.repo_name:
        print("Please provide your Hugging Face username to continue.")
        username = input("Hugging Face username: ").strip()
        repo_name = f"{username}/{config['huggingface']['repo_name']}"
    else:
        repo_name = args.repo_name
    
    model_card = create_model_card(config, repo_name)
    model_card_path = Path(args.model_path) / "README.md"
    
    print("Creating model card...")
    with open(model_card_path, "w") as f:
        f.write(model_card)
    
    if not args.token:
        print("\nTo upload to Hugging Face, you need an API token.")
        print("Get your token from: https://huggingface.co/settings/tokens")
        token = input("Hugging Face API token (or press Enter to skip upload): ").strip()
        
        if not token:
            print("\nModel card created locally. To upload later, run:")
            print(f"python upload_model.py --repo-name {repo_name} --token YOUR_TOKEN")
            return
    else:
        token = args.token
    
    upload_to_huggingface(
        args.model_path, 
        repo_name, 
        token, 
        args.private or config['huggingface']['private']
    )
    
    print("\n✨ Upload complete! ✨")
    print(f"Your model is now available at: https://huggingface.co/{repo_name}")
    print("\nTo use your model:")
    print(f"model = PeftModel.from_pretrained(base_model, '{repo_name}')")

if __name__ == "__main__":
    main()