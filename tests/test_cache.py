import pytest
import os
from mi_datasets.core.cache import CacheManager

def test_cache_manager_lifecycle(tmp_path):
    base_dir = str(tmp_path / "mi_cache")
    manager = CacheManager(base_dir=base_dir)
    assert os.path.exists(base_dir)

    identifier = "vision/mock_dataset"
    dataset_dir = manager.get_dataset_dir(identifier)
    assert os.path.exists(dataset_dir)
    assert dataset_dir.endswith("vision/mock_dataset")

    assert not manager.is_cached(dataset_dir)
    
    dummy_file = os.path.join(dataset_dir, "data.pt")
    with open(dummy_file, "w") as f:
        f.write("mock_data")
        
    assert manager.is_cached(dataset_dir)
    assert manager.is_cached(dataset_dir, required_files=["data.pt"])
    assert not manager.is_cached(dataset_dir, required_files=["missing.pt"])

    manager.clear_cache(identifier)
    assert not os.path.exists(dataset_dir)
    assert os.path.exists(base_dir)

    manager.clear_cache()
    assert os.path.exists(base_dir)
    assert len(os.listdir(base_dir)) == 0