#!/usr/bin/env python3
import json
import os
from datasets import Dataset, DatasetDict, load_dataset
from typing import List, Dict
import random

def load_kawaii_dataset(path: str) -> List[Dict[str, str]]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_conversation(instruction: str, response: str, system_prompt: str = None) -> str:
    if system_prompt:
        return f"<|system|>{system_prompt}<|end|><|user|>{instruction}<|end|><|assistant|>{response}<|end|>"
    return f"<|user|>{instruction}<|end|><|assistant|>{response}<|end|>"

def prepare_anime_quotes():
    print("Loading anime quotes dataset from HuggingFace...")
    dataset = load_dataset("sarthak-2002/anime-quotes", split="train")
    
    formatted_data = []
    for item in dataset:
        if 'quote' in item and item['quote']:
            instruction = "Share an inspiring anime quote"
            response = f"{item['quote']} - {item.get('character', 'Unknown')}, {item.get('anime', 'Unknown')}"
            formatted_data.append({
                "text": format_conversation(
                    instruction,
                    f"Here's a beautiful quote for you, senpai~ {response} Isn't it wonderful? It reminds me why I love anime so much!"
                )
            })
    
    return formatted_data

def prepare_kawaii_dataset():
    print("Loading kawaii responses dataset...")
    kawaii_data = load_kawaii_dataset("data/kawaii_responses.json")
    
    system_prompt = "You are Mikasa, a cute and kawaii AI assistant. You love your senpai (the user) and express yourself in an enthusiastic, slightly tsundere manner. You use Japanese honorifics and expressions naturally. You're helpful, protective, and always eager to assist your senpai."
    
    formatted_data = []
    for item in kawaii_data:
        formatted_data.append({
            "text": format_conversation(
                item["instruction"],
                item["response"],
                system_prompt
            )
        })
    
    return formatted_data

def combine_datasets(anime_data, kawaii_data, train_split=0.9):
    all_data = anime_data + kawaii_data
    
    random.shuffle(all_data)
    
    split_idx = int(len(all_data) * train_split)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    return train_data, val_data

def main():
    os.makedirs("data/processed", exist_ok=True)
    
    print("Preparing anime quotes dataset...")
    anime_data = prepare_anime_quotes()
    print(f"Loaded {len(anime_data)} anime quotes")
    
    print("\nPreparing kawaii responses dataset...")
    kawaii_data = prepare_kawaii_dataset()
    print(f"Loaded {len(kawaii_data)} kawaii responses")
    
    print("\nCombining and splitting datasets...")
    train_data, val_data = combine_datasets(anime_data, kawaii_data)
    
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })
    
    print("\nSaving processed dataset...")
    dataset_dict.save_to_disk("data/processed/mikasa_dataset")
    
    with open("data/processed/train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open("data/processed/val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    print("\nDataset preparation complete!")
    print("Files saved to data/processed/")
    
    print("\nSample training examples:")
    for i in range(min(3, len(train_data))):
        print(f"\nExample {i+1}:")
        print(train_data[i]['text'][:200] + "...")

if __name__ == "__main__":
    main()