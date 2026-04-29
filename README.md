# 🔬 Mechanistic Interpretability Datasets (`mi_datasets`)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Coverage](https://img.shields.io/badge/coverage-88.0%25-yellow)

A dataset harness designed specifically for my **Mechanistic Interpretability (MI)** research. Basically, I just wanted a standardized way of interacting with the datasets I regularly use. This repository is heavily tailored to my personal needs, but might be useful for others also.

For MI, exact reproducibility is important. Standard data loaders and transform pipelines often discard critical spatial parameters, silently mutate data schemas, or lack aggressive caching. `mi_datasets` solves this by providing **strictly typed data contracts**, **serialized transformation pipelines**, and **deterministic caching**, ensuring exact alignment between network inputs and internal activations.

## ✨ Key Features

* **Strict Data Contracts:** Every provider yields a strictly typed `DataItem` (and `DataBatch`). You always receive both the `raw_input` (for visualization/feature attribution) and `model_input` (the processed tensor).
* **Transform Serialization:** Never lose track of how a tensor was normalized or cropped. Pipelines are parsed and serialized (`dataset.get_info()`) to map activations back to raw spatial/graph structures. For each sample, we also get the detailed transformation history for inversion.
* **Deterministic Subsampling:** `subset_fraction` and `seed` configurations allow for exactly reproducible ablation studies without loading the entire dataset into RAM.
* **Decorator-Based Registry:** Add new datasets anywhere in your project without modifying the core repository. 
* **Global Caching:** Sane defaults with `MI_DATASETS_CACHE` environment variable support for cluster environments.

---

## 📦 Installation

```bash
git clone [https://github.com/your-org/mi_datasets.git](https://github.com/your-org/mi_datasets.git)
cd mi_datasets
# Create a virtual environment and install the package in editable mode
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Optional: Set Cache Directory

Set your global cache directory
```bash
export MI_DATASETS_CACHE="path/to/cache/dir"
```
Defaults to `~/.cache/mi_datasets`.

## :lightning: Quickstart

```python
from mi_datasets.core.registry import load_dataset
from mi_datasets.modalities.vision.transforms import (
    TrackedCompose, 
    TrackedRandomCrop, 
    TrackedNormalize,
    TrackedToTensor
)

# 1. Define transform
pipeline = TrackedCompose([
    TrackedRandomCrop(size=(24, 24)),
    TrackedToTensor(),
    TrackedNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# 2. Load the dataset
dataset = load_dataset(
    "torchvision/cifar10", 
    split="train", 
    subset_fraction=0.01,
    transform=pipeline,
)

# 3. Access strictly typed data
item = dataset[0]
print(item.model_input.shape)  # The tensor for the model
print(type(item.raw_input))    # The raw PIL Image for visualization
```

## Components

```
mi_datasets/
├── core/
│   ├── __init__.py
│   ├── base.py            # Base Class
│   ├── registry.py        # Decorator-based registration & factory system
│   ├── cache.py           # Unified caching mechanism
│   └── types.py           # Dataclass schemas
├── providers/
│   ├── __init__.py
│   ├── torchvision.py     # Torchvision Datasets
│   ├── huggingface.py     # Huggingface Datasets Wrapper
│   ├── torch_geometric.py # [Under Construction]
│   └── web_url.py         # [Under Construction]
├── modalities/
│   ├── __init__.py
│   ├── vision/
│   │   ├── transforms.py  # Standardized image preprocessing
│   │   └── collate.py     # Vision-specific batching
│   └── graph/
│       ├── transforms.py  # [Under Construction]
│       └── collate.py     # [Under Construction]
├── configs/               # Standardized YAML configs
├── tests/
├── pyproject.toml
└── README.md
```