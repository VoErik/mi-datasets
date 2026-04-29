import pytest
import torch
import os
from typing import Tuple, Dict, Any
from mi_datasets.core.base import BaseMIDataset
from mi_datasets.core.type import DataItem, DataBatch, DatasetInfo


class DummyDataset(BaseMIDataset):
    """A concrete implementation of the Base ABC for testing purposes."""
    _modality = "dummy_modality"

    def __init__(self, config: dict, **kwargs):
        self.download_called = False
        self.load_called = False
        super().__init__(config, **kwargs)

    def _is_cached(self) -> bool:
        return self.config.get("mock_is_cached", True)

    def _download(self) -> None:
        self.download_called = True

    def _load_into_memory(self) -> None:
        self.load_called = True

    def _get_full_length(self) -> int:
        return 100

    def _get_raw_data(self, idx: int) -> dict:
        return {
            "id": f"item_{idx}",
            "input": torch.tensor([idx, idx], dtype=torch.float32),
            "target": torch.tensor(idx),
            "metadata": {"custom_meta": "exists"}
        }

def standard_transform(x: torch.Tensor) -> torch.Tensor:
    return x * 2

def mi_tracked_transform(x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
    return x * 3, {"name": "TrackedMock", "params": {"p": 1}}

class SerializableTransform:
    """Tests the .to_dict() parsing branch."""
    def to_dict(self):
        return {"param_a": 42}
    def __call__(self, x): return x

class DictFallbackTransform:
    """Tests the __dict__ fallback parsing branch."""
    def __init__(self):
        self.param_b = 99
        self._private = "hidden"
    def __call__(self, x): return x

class MockCompose:
    """Tests the Compose/Pipeline parsing branch."""
    def __init__(self):
        self.transforms = [SerializableTransform(), DictFallbackTransform()]


def test_base_setup_and_cache_lifecycle(tmp_path):
    cache_dir = str(tmp_path / "mock_cache")
    
    ds_not_cached = DummyDataset({"mock_is_cached": False}, cache_dir=cache_dir)
    assert ds_not_cached.download_called is True
    assert ds_not_cached.load_called is True
    assert os.path.exists(cache_dir)

    ds_cached = DummyDataset({"mock_is_cached": True}, cache_dir=cache_dir)
    assert ds_cached.download_called is False
    assert ds_cached.load_called is True

def test_subset_generation():
    ds_full = DummyDataset({"subset_fraction": 1.0})
    assert len(ds_full) == 100

    ds_partial = DummyDataset({"subset_fraction": 0.3, "seed": 42})
    assert len(ds_partial) == 30
    
    indices = ds_partial._indices
    assert indices == sorted(indices), "Subset indices are not sorted!"

def test_getitem_transform_branching():
    ds_none = DummyDataset({})
    item1 = ds_none[0]
    assert item1.model_input[0].item() == 0  # 0 * 1

    ds_std = DummyDataset({}, transform=standard_transform)
    item2 = ds_std[1]
    assert item2.model_input[0].item() == 2  # 1 * 2
    assert "transform_history" not in item2.metadata

    # Test 3: MI Tuple-returning transform (Tracks history)
    ds_mi = DummyDataset({}, transform=mi_tracked_transform, target_transform=lambda x: x + 10)
    item3 = ds_mi[2]
    assert item3.model_input[0].item() == 6  # 2 * 3
    assert item3.targets.item() == 12        # 2 + 10
    assert "transform_history" in item3.metadata
    assert item3.metadata["transform_history"]["name"] == "TrackedMock"

def test_default_collate_and_dataloader():
    ds = DummyDataset({})
    batch_list = [ds[0], ds[1], ds[2]]
    
    collate_fn = ds.get_collate_fn()
    batch = collate_fn(batch_list)
    
    assert isinstance(batch, DataBatch)
    assert batch.model_inputs.shape == (3, 2)
    assert batch.targets.shape == (3,)
    assert len(batch.ids) == 3

    dl = ds.get_dataloader(batch_size=2, shuffle=False)
    assert dl.batch_size == 2

def test_parse_transforms_and_get_info():
    config = {"_identifier": "dummy_modality/mock_data"}
    ds = DummyDataset(
        config=config, 
        transform=MockCompose(), 
        target_transform=SerializableTransform()
    )
    
    info = ds.get_info()
    
    assert isinstance(info, DatasetInfo)
    assert info.modality == "dummy_modality"
    assert info.name == "mock_data"
    assert info.num_items == 100
    
    assert len(info.transforms) == 2
    assert info.transforms[0].name == "SerializableTransform"
    assert info.transforms[0].params["param_a"] == 42
    
    assert info.transforms[1].name == "DictFallbackTransform"
    assert info.transforms[1].params["param_b"] == 99
    assert "_private" not in info.transforms[1].params
    
    assert len(info.target_transforms) == 1