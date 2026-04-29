import pytest
from mi_datasets import load_dataset

def test_subset_determinism():
    config_seed_42 = {"split": "train", "subset_fraction": 0.05, "seed": 42}
    config_seed_99 = {"split": "train", "subset_fraction": 0.05, "seed": 99}

    ds_1 = load_dataset("torchvision/cifar10", config=config_seed_42)
    ds_2 = load_dataset("torchvision/cifar10", config=config_seed_42)
    
    ds_3 = load_dataset("torchvision/cifar10", config=config_seed_99)

    indices_1 = ds_1._indices
    indices_2 = ds_2._indices
    indices_3 = ds_3._indices

    assert indices_1 == indices_2, "Identical seeds produced divergent subsets."
    assert indices_1 != indices_3, "Different seeds produced identical subsets."
    
    assert indices_1 == sorted(indices_1), "Subset indices are not sorted, threatening I/O performance."