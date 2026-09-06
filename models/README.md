# 🧠 Daphne Model Hub & Weights Index

> **"Where the weights live, rent-free, until your RAM files for eviction."**

This directory is the local warehouse for supported models, weights, configs, and tokenizers. 

`models/README.md` acts as our index of targets, sources, and download cheat-sheets.

---

## 📋 Supported & Target Models

| Model | Architecture | Size / Quant | HuggingFace Source | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Gemma 4 26B-A4B-it** | Gemma 4 (Decoder MoE / Hybrid) | ~26B total (4B active) | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) | 🎯 Primary Target |
| *More to come...* | *TBD* | *TBD* | *Watch this space* | ⏳ Soon™ |

---

## 📥 How to Download & Organize Weights

We use **Hugging Face** and its CLI tool (`hf` / `huggingface-cli`) to grab weights because it makes downloading, resuming, and versioning painless.

Daphne expects model weights to live right inside `models/weights/<model-id>/` with their official Safetensors and JSON configs intact.

### 1. Gemma 4 (`gemma-4-26B-A4B-it`)

- **Model Page & License:** [Hugging Face — google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)
- **Architecture Notes:** Fast 4B active parameters dynamic routing on top of a 26B parameter sparse backbone.

Quick download command using `hf`:

```bash
hf download google/gemma-4-26B-A4B-it --local-dir models/weights/gemma-4-26B-A4B-it
```

---

## 🗂️ Expected Directory Layout

Each model directory under `models/weights/` should look like this:

```text
models/
├── README.md
└── weights/
    └── gemma-4-26B-A4B-it/
        ├── config.json                     # Hidden dimensions, layer count, head count, RoPE base
        ├── tokenizer.json                  # Vocabulary and merge rules
        ├── tokenizer_config.json           # Special tokens (<start_of_turn>, <end_of_turn>)
        ├── chat_template.jinja             # Prompt formatting template
        ├── model.safetensors.index.json    # Shard index (for multi-file weights)
        └── *.safetensors                   # Raw binary tensor weights
```

---

## ⚠️ Notes on Disk Space & Git

> [!WARNING]
> Safetensors files are massive (gigabytes to tens of gigabytes). 
> **Never commit `.safetensors`, `.bin`, or `.pt` files to git.** Keep your `.gitignore` sharp, or your GitHub pushes will be banished to the shadow realm.
