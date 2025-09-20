#!/usr/bin/env python3
import os
import sys
import torch
import yaml
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer
import json
import platform



def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_model_and_tokenizer(config):
    print(f"Loading model: {config['model']['name']}")

    # Detect if we're on Mac
    is_mac = platform.system() == "Darwin"

    model_kwargs = {
        "trust_remote_code": True,
    }

    if is_mac:
        # Mac-specific settings using MPS
        print("Detected Mac - using MPS (Metal Performance Shaders)")
        if torch.backends.mps.is_available():
            device_map = {"": "mps"}
            model_kwargs["dtype"] = torch.float16
        else:
            print("MPS not available, using CPU")
            device_map = {"": "cpu"}
            model_kwargs["dtype"] = torch.float32
    else:
        # For Linux/Windows with CUDA
        if torch.cuda.is_available():
            device_map = "auto"
            model_kwargs["dtype"] = torch.float16
            # Only use quantization on CUDA systems
            if config['model'].get('load_in_8bit'):
                model_kwargs["load_in_8bit"] = True
        else:
            device_map = {"": "cpu"}
            model_kwargs["dtype"] = torch.float32

    model_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        **model_kwargs
    )

    # Enable gradient checkpointing to save memory when requested
    if config['training'].get('gradient_checkpointing'):
        model.gradient_checkpointing_enable()
        # Latest Transformers recommend disabling cache for training when using GC
        if hasattr(model, 'config'):
            model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name'],
        trust_remote_code=True,
        padding_side="right",
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

def setup_lora(model, config):
    print("Setting up LoRA configuration...")

    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        task_type=getattr(TaskType, config['lora']['task_type']),
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model

def load_dataset_from_json(train_path, val_path):
    print("Loading datasets...")

    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    with open(val_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    return train_dataset, val_dataset

def setup_training_args(config):
    training_config = config['training']

    # Some Transformers versions use eval_strategy instead of evaluation_strategy.
    # We map dynamically based on TrainingArguments fields.
    try:
        from dataclasses import fields as dataclass_fields
        ta_field_names = {f.name for f in dataclass_fields(TrainingArguments)}
    except Exception:
        ta_field_names = set()
    eval_key = 'evaluation_strategy' if 'evaluation_strategy' in ta_field_names else 'eval_strategy'

    # Coerce numeric types defensively to avoid string values from configs
    def _to_int(x, default=0):
        try:
            return int(x)
        except Exception:
            return default
    def _to_float(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return default
    kwargs = dict(
        output_dir=str(training_config['output_dir']),
        num_train_epochs=_to_float(training_config.get('num_train_epochs', 1)),
        per_device_train_batch_size=_to_int(training_config.get('per_device_train_batch_size', 1)),
        per_device_eval_batch_size=_to_int(training_config.get('per_device_eval_batch_size', 1)),
        gradient_accumulation_steps=_to_int(training_config.get('gradient_accumulation_steps', 1)),
        learning_rate=_to_float(training_config.get('learning_rate', 5e-5)),
        logging_steps=_to_int(training_config.get('logging_steps', 10)),
        save_strategy=str(training_config.get('save_strategy', 'steps')),
        save_steps=_to_int(training_config.get('save_steps', 500)),
        eval_steps=_to_int(training_config.get('eval_steps', 500)),
        optim=str(training_config.get('optim', 'adamw_torch')),
        report_to=str(training_config.get('report_to', 'none')),
        gradient_checkpointing=bool(training_config.get('gradient_checkpointing', False)),
        remove_unused_columns=False,
    )
    kwargs[eval_key] = training_config['evaluation_strategy']

    return TrainingArguments(**kwargs)

def main():
    config = load_config()

    print("=" * 50)
    print("MIKASA TRAINING INITIALIZED")
    print(f"Platform: {platform.system()}")
    print(f"PyTorch version: {torch.__version__}")
    if platform.system() == "Darwin":
        print(f"MPS available: {torch.backends.mps.is_available()}")
    else:
        print(f"CUDA available: {torch.cuda.is_available()}")
    print("=" * 50)

    model, tokenizer = setup_model_and_tokenizer(config)

    model = setup_lora(model, config)

    train_dataset, val_dataset = load_dataset_from_json(
        config['dataset']['train_path'],
        config['dataset']['val_path']
    )

    training_args = setup_training_args(config)

    print("\nStarting training...")
    # Build SFTTrainer kwargs based on installed TRL version
    from inspect import signature as _sig
    _params = set(_sig(SFTTrainer.__init__).parameters.keys())
    sft_kwargs = {
        'model': model,
        'args': training_args,
        'train_dataset': train_dataset,
        'eval_dataset': val_dataset,
    }
    if 'tokenizer' in _params:
        sft_kwargs['tokenizer'] = tokenizer
    if 'dataset_text_field' in _params:
        sft_kwargs['dataset_text_field'] = config['dataset']['text_field']
    if 'max_seq_length' in _params:
        ms = config['training'].get('max_seq_length', None)
        if ms is not None:
            try:
                ms = int(ms)
            except Exception:
                ms = None
        sft_kwargs['max_seq_length'] = ms
    if 'packing' in _params:
        sft_kwargs['packing'] = bool(config['training'].get('packing', False))

    trainer = SFTTrainer(**sft_kwargs)

    trainer.train()

    print("\nSaving final model...")
    trainer.save_model()

    output_dir = config['training']['output_dir']
    tokenizer.save_pretrained(output_dir)

    with open(f"{output_dir}/config.yaml", "w") as f:
        yaml.dump(config, f)

    print(f"\nTraining complete! Model saved to {output_dir}")
    print("\nTo test the model, run: python chat_with_mikasa.py")

if __name__ == "__main__":
    main()