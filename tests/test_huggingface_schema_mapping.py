import pytest
from mi_datasets import load_dataset

def test_huggingface_schema_mapping():
    config = {
        "path": "rotten_tomatoes",
        "split": "train",
        "subset_fraction": 0.01,
        "input_col": "text",      # Maps HF 'text' to model_input
        "target_col": "label",    # Maps HF 'label' to targets
    }

    ds = load_dataset("huggingface", config=config)
    
    assert len(ds) > 0, "Dataset failed to load."
    
    item = ds[0]
    
    assert hasattr(item, "id")
    assert hasattr(item, "model_input")
    assert hasattr(item, "targets")
    assert hasattr(item, "metadata")
    
    assert isinstance(item.model_input, str), "Input column mapping failed."
    assert isinstance(item.targets, int), "Target column mapping failed."