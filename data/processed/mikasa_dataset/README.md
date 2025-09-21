---
language:
- en
pretty_name: Mikasa Kawaii AI Assistant Dataset
tags:
- conversational
- fine-tuning
- anime
- kawaii
- chatbot
- instruction-following
license: gpl-3.0
task_categories:
- text-generation
size_categories:
- n<1K
---

# 🌸 Mikasa Kawaii AI Assistant Dataset

![Mikasa](mikasa.png)

## Dataset Description

This dataset contains curated conversation pairs designed to fine-tune language models to behave like Mikasa - a devoted, kawaii AI assistant with anime-inspired personality traits. The dataset features Japanese honorifics, tsundere characteristics, and enthusiastic helping behavior.

### Dataset Summary

- **Language:** English with Japanese honorifics
- **Task:** Conversational AI / Instruction Following
- **Size:** 48 conversation pairs (43 train, 5 validation)
- **Format:** Chat template with system/user/assistant roles
- **Base Model Compatibility:** Optimized for Qwen/Qwen3-4B models

## 📊 Dataset Structure

### Data Fields

- **text** (string): Complete conversation in chat template format including:
  - System prompt defining Mikasa's personality
  - User query/instruction
  - Assistant response in character

### Data Splits

| Split | Examples |
|-------|----------|
| train | 43 |
| validation | 5 |

### Example

```json
{
  "text": "<|system|>You are Mikasa, a cute and kawaii AI assistant...<|end|><|user|>Hello!<|end|><|assistant|>Ohayo, senpai! I'm so happy to see you~...<|end|>"
}
```

## 🎯 Intended Use

This dataset is designed for:
- Fine-tuning language models for anime-style personality
- Creating engaging conversational AI assistants
- Research in personality-driven AI responses
- Entertainment and educational chatbot development

### Usage with Transformers

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("RamboRogers/mikasa-dataset")

# Access training data
train_data = dataset["train"]
```

## 📝 Dataset Creation

### Source Data

The dataset combines:
1. Custom-written conversational pairs emphasizing kawaii personality traits
2. Curated responses incorporating Japanese honorifics and expressions
3. Technical assistance examples with character-consistent responses

### Curation Process

Each conversation was carefully crafted to:
- Maintain consistent personality across diverse topics
- Balance helpfulness with entertaining character traits
- Include appropriate Japanese cultural elements
- Cover various assistance scenarios (coding, general knowledge, casual chat)

## 💡 Considerations

### Limitations

- Small dataset size (ideal for LoRA/QLoRA fine-tuning)
- English-primary with Japanese elements (not for Japanese language tasks)
- Personality-focused rather than knowledge-focused

### Ethical Considerations

- Dataset promotes positive, helpful interactions
- No harmful, toxic, or inappropriate content
- Respectful use of Japanese cultural elements
- Designed for entertainment and assistance, not deception

## 📄 Licensing

This dataset is released under **GNU General Public License v3.0 (GPLv3)**.

### Citation

If you use this dataset, please cite:

```bibtex
@dataset{mikasa_dataset_2024,
  author = {Matthew Rogers},
  title = {Mikasa Kawaii AI Assistant Dataset},
  year = {2024},
  publisher = {HuggingFace},
  url = {https://huggingface.co/datasets/RamboRogers/mikasa-dataset}
}
```

## 🔗 Additional Information

### Dataset Curated by
**Matthew Rogers (RamboRogers)**

[![GitHub](https://img.shields.io/badge/GitHub-RamboRogers-181717?style=for-the-badge&logo=github)](https://github.com/RamboRogers)
[![Twitter](https://img.shields.io/badge/Twitter-@rogerscissp-1DA1F2?style=for-the-badge&logo=twitter)](https://x.com/rogerscissp)
[![Website](https://img.shields.io/badge/Web-matthewrogers.org-00ADD8?style=for-the-badge&logo=google-chrome)](https://matthewrogers.org)

### Related Resources

- **Model Repository:** [RamboRogers/mikasa-ft](https://huggingface.co/RamboRogers/mikasa-ft)
- **GitHub Project:** [github.com/RamboRogers/mikasa](https://github.com/RamboRogers/mikasa)
- **Base Model:** [Qwen/Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)

### Contact

For questions, feedback, or collaboration:
- Open an issue on [GitHub](https://github.com/RamboRogers/mikasa)
- Reach out on [Twitter/X](https://x.com/rogerscissp)
- Visit [matthewrogers.org](https://matthewrogers.org)

---

<div align="center">

**Made with 💕 by RamboRogers**

*Building kawaii AI, one dataset at a time~*

</div>