import pytest
import torch
import torchvision.transforms.functional as F
from mi_datasets.core.type import DataItem
from mi_datasets.modalities.vision.transforms import (
    TrackedCenterCrop,
    TrackedRandomCrop,
    TrackedResize,
    TrackedNormalize,
    TrackedRandomRotation,
    TrackedCompose
)

@pytest.fixture
def dummy_image_item():
    """Provides a standard 3x32x32 dummy image wrapped in a DataItem."""
    img = torch.arange(32*32, dtype=torch.float32).view(1, 32, 32).expand(3, -1, -1)
    
    return DataItem(
        id=0,
        model_input=img.clone(),
        raw_input=img.clone(),
        targets=1,
        metadata={}
    )

def test_tracked_center_crop(dummy_image_item):
    transform = TrackedCenterCrop(size=(16, 16))
    
    result_tensor, params = transform(dummy_image_item.model_input)
    
    assert result_tensor.shape == (3, 16, 16), "Crop failed to resize tensor."
    
    assert params["crop_top"] == 8  # (32 - 16) / 2
    assert params["crop_left"] == 8
    assert params["orig_h"] == 32

    dummy_activation = torch.ones(1, 16, 16)
    inverse_mask = transform.inverse(dummy_activation, params)
    
    assert inverse_mask.shape == (1, 32, 32), "Inverse mapping returned incorrect dimensions."
    assert inverse_mask[0, 0, 0] == 0, "Outside crop area should be padded with zeros."
    assert inverse_mask[0, 8, 8] == 1, "Inside crop area should retain activation values."

def test_tracked_random_crop(dummy_image_item):
    torch.manual_seed(42)
    transform = TrackedRandomCrop(size=(20, 20))
    
    result_tensor, params = transform(dummy_image_item.model_input)
    
    assert result_tensor.shape == (3, 20, 20)
    assert "crop_top" in params
    assert "crop_left" in params
    
    dummy_activation = torch.ones(1, 20, 20)
    inverse_mask = transform.inverse(dummy_activation, params)
    assert inverse_mask.shape == (1, 32, 32)

def test_tracked_resize(dummy_image_item):
    transform = TrackedResize(size=(64, 64))
    
    result_tensor, params = transform(dummy_image_item.model_input)
    
    assert result_tensor.shape == (3, 64, 64)
    assert params["scale_factor_h"] == 2.0
    assert params["scale_factor_w"] == 2.0
    
    dummy_activation = torch.ones(1, 64, 64)
    inverse_mask = transform.inverse(dummy_activation, params)
    assert inverse_mask.shape == (1, 32, 32)

def test_tracked_normalize(dummy_image_item):
    mean, std = (0.5, 0.5, 0.5), (0.2, 0.2, 0.2)
    transform = TrackedNormalize(mean=mean, std=std)
    
    controlled_tensor = torch.ones(3, 32, 32) * 0.7 
    result_tensor, params = transform(controlled_tensor)
    
    assert torch.allclose(result_tensor, torch.ones(3, 32, 32)), "Forward normalization failed math."
    assert params["mean"] == mean
    
    inverse_tensor = transform.inverse(result_tensor, params)
    assert torch.allclose(inverse_tensor, controlled_tensor), "Inverse un-normalization failed math."

def test_tracked_random_rotation(dummy_image_item):
    torch.manual_seed(42)
    transform = TrackedRandomRotation(degrees=45.0)
    
    result_tensor, params = transform(dummy_image_item.model_input)
    
    assert result_tensor.shape == (3, 32, 32)
    assert -45.0 <= params["angle"] <= 45.0
    
    inverse_tensor = transform.inverse(result_tensor, params)
    assert inverse_tensor.shape == (3, 32, 32)

def test_tracked_compose(dummy_image_item):
    transform_chain = TrackedCompose([
        TrackedCenterCrop(size=(16, 16)),
        TrackedNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    result_tensor, history = transform_chain(dummy_image_item.model_input)
    
    assert result_tensor.shape == (3, 16, 16)
    assert len(history) == 2
    assert history[0]["name"] == "TrackedCenterCrop"
    assert history[1]["name"] == "TrackedNormalize"
    
    inverse_chain_tensor = transform_chain.inverse(result_tensor, history)
    
    assert inverse_chain_tensor.shape == (3, 32, 32)