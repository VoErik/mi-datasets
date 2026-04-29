import pytest
import torch
from mi_datasets.core.type import DataItem, DataBatch, TransformMeta, DatasetInfo

def test_databatch_to_device():
    ids = [1, 2]
    model_inputs = torch.randn(2, 3, 16, 16)
    raw_inputs = ["image_1.png", "image_2.png"]
    targets = torch.tensor([0, 1])
    
    batch = DataBatch(
        ids=ids,
        model_inputs=model_inputs,
        raw_inputs=raw_inputs,
        targets=targets,
        metadata=[{"meta": 1}, {"meta": 2}]
    )
    
    target_device = torch.device("cpu")
    moved_batch = batch.to(target_device)
    
    assert isinstance(moved_batch, DataBatch)
    assert moved_batch.model_inputs.device == target_device
    assert moved_batch.targets.device == target_device
    assert moved_batch.raw_inputs == raw_inputs
    assert moved_batch.ids == ids

def test_databatch_to_device_with_dict_targets():
    model_inputs = torch.randn(2, 3, 16, 16)
    targets_dict = {
        "labels": torch.tensor([0, 1]),
        "masks": torch.randn(2, 1, 16, 16)
    }
    
    batch = DataBatch(
        ids=[1, 2],
        model_inputs=model_inputs,
        raw_inputs=["raw1", "raw2"],
        targets=targets_dict
    )
    
    target_device = torch.device("cpu")
    moved_batch = batch.to(target_device)
    
    assert isinstance(moved_batch.targets, dict)
    assert moved_batch.targets["labels"].device == target_device
    assert moved_batch.targets["masks"].device == target_device

def test_dataset_info_string_formatting():
    long_list = ["class_0", "class_1", "class_2", "class_3", "class_4", "class_5"]
    
    info = DatasetInfo(
        name="test_dataset",
        modality="vision",
        provider="TestProvider",
        split="train",
        num_items=1000,
        features={"image": "Tensor", "label": "int"},
        transforms=[
            TransformMeta(name="TrackedResize", params={"size": 32}, is_spatial=True)
        ],
        target_transforms=[],
        metadata={"classes": long_list, "num_classes": 6}
    )
    
    info_str = str(info)
    
    assert "Dataset: TEST_DATASET [vision]" in info_str
    assert "Split: train | Total Items: 1000" in info_str
    assert "• image: Tensor" in info_str
    assert "• TrackedResize (Spatial: True)" in info_str
    assert "size: 32" in info_str
    
    assert "len=6" in info_str
    assert "[class_0, class_1, ..., class_5]" in info_str
    
    assert repr(info) == info_str