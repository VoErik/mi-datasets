import pytest
from mi_datasets import load_dataset

def test_cifar10_initialization():
    ds = load_dataset("torchvision/cifar10", config={"split": "train", "subset_fraction": 0.01})
    
    assert len(ds) == 500, "1% of 50000 should be 500"
    
    item = ds[0]
    assert "class_name" in item.metadata
    assert item.model_input is not None

def test_registry_throws_on_invalid():
    with pytest.raises(KeyError):
        load_dataset("torchvision/does_not_exist")