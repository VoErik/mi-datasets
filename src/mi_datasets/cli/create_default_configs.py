from pathlib import Path

TEMPLATES = {
    "torchvision_cifar10.yaml": """# Configuration for Torchvision CIFAR-10
dataset: "torchvision/cifar10"

# Dataset parameters
split: "train"               # Options: 'train', 'test'
subset_fraction: 1.0         # Float between 0.0 and 1.0 for deterministic subsetting
seed: 42                     # Seed for the subset RNG
keep_raw_input: true         # Keeps the PIL Image (set to false for training)
keep_metadata: true          # Keeps metadata (e.g., if you need transform_history for custom targets)

# Environment overrides (Optional)
# cache_dir: "~/.cache/mi_datasets/custom_run"
""",
"torchvision_cifar100.yaml": """# Configuration for Torchvision CIFAR-100
dataset: "torchvision/cifar100"

# Dataset parameters
split: "train"               # Options: 'train', 'test'
subset_fraction: 1.0         # Float between 0.0 and 1.0 for deterministic subsetting
seed: 42                     # Seed for the subset RNG
keep_raw_input: true         # Keeps the PIL Image (set to false for training)
keep_metadata: true          # Keeps metadata (e.g., if you need transform_history for custom targets)

# Environment overrides (Optional)
# cache_dir: "~/.cache/mi_datasets/custom_run"
""",

    "huggingface_text.yaml": """# Configuration for arbitrary HuggingFace datasets
dataset: "huggingface"

# HuggingFace Hub routing
path: "wikitext"             # The HuggingFace repository ID
name: "wikitext-2-raw-v1"    # The subset/config name (remove if not applicable)
split: "train"               # Usually 'train', 'validation', or 'test'

# Schema mapping (Critical for mapping arbitrary datasets to the MI schema)
input_col: "text"            # Mapped to DataItem.model_input
target_col: null             # Mapped to DataItem.targets (leave null for self-supervised)

subset_fraction: 1.0
seed: 42
keep_raw_input: true         # Keeps the PIL Image (set to false for training)
keep_metadata: true          # Keeps metadata (e.g., if you need transform_history for custom targets)
""",

    "torchvision_celeba.yaml": """# Configuration for Torchvision CelebA
dataset: "torchvision/celeba"

split: "train"               # Options: 'train', 'valid', 'test', 'all'
target_type: "attr"          # Options: 'attr', 'identity', 'bbox', 'landmarks'

subset_fraction: 1.0
seed: 42
keep_raw_input: true         # Keeps the PIL Image (set to false for training)
keep_metadata: true          # Keeps metadata (e.g., if you need transform_history for custom targets)
"""
}

def init_configs():
    """Generates default YAML configuration templates in the configs/ directory."""
    target_dir = Path("configs")
    target_dir.mkdir(exist_ok=True)
    
    print(f"Initializing default configurations in ./{target_dir}...")
    
    for filename, content in TEMPLATES.items():
        filepath = target_dir / filename
        if filepath.exists():
            print(f"  [SKIP] {filename} already exists.")
            continue
            
        with open(filepath, "w") as f:
            f.write(content)
        print(f"  [CREATE] {filename}")
        
    print("Configuration generation complete.")

if __name__ == "__main__":
    init_configs()