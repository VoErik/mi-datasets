import torch
import matplotlib.pyplot as plt
from typing import List, Optional
import torchvision.transforms.functional as F
from mi_datasets.core.type import DataItem, DataBatch
from mi_datasets.modalities.vision.transforms import TrackedCompose

def plot_item_with_inverse(item: DataItem, pipeline: TrackedCompose):
    """
    Plots the raw image, the model-ready tensor, and the inverted tensor
    to prove the mechanistic tracking works.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(item.raw_input)
    axes[0].set_title(f"Raw Input (ID: {item.id})")
    axes[0].axis("off")

    tensor_img = item.model_input.detach().cpu()

    viz_tensor = (tensor_img - tensor_img.min()) / (tensor_img.max() - tensor_img.min())
    axes[1].imshow(viz_tensor.permute(1, 2, 0))
    axes[1].set_title("Model Input (Transformed)")
    axes[1].axis("off")

    history = item.metadata.get("transform_history")
    if history:
        inverted_tensor = pipeline.inverse(item.model_input, history)
        
        inverted_viz = inverted_tensor.detach().cpu().permute(1, 2, 0)
        inverted_viz = torch.clamp(inverted_viz, 0, 1)
        
        axes[2].imshow(inverted_viz)
        axes[2].set_title("Inverted (Mapped Back)")
    else:
        axes[2].text(0.5, 0.5, "No Transform History", ha='center', va='center')
        axes[2].set_title("Inverted (Failed)")
        
    axes[2].axis("off")
    plt.tight_layout()
    plt.show()

def plot_batch(batch: DataBatch, max_items: int = 4):
    """
    Plots a grid of raw inputs vs model inputs for a collated batch.
    """
    num_items = min(len(batch.ids), max_items)
    fig, axes = plt.subplots(2, num_items, figsize=(3 * num_items, 6))
    
    if num_items == 1:
        axes = axes.reshape(2, 1)

    for i in range(num_items):
        axes[0, i].imshow(batch.raw_inputs[i])
        axes[0, i].set_title(f"Raw ID: {batch.ids[i]}")
        axes[0, i].axis("off")

        tensor = batch.model_inputs[i].detach().cpu()
        viz_tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        axes[1, i].imshow(viz_tensor.permute(1, 2, 0))
        axes[1, i].set_title(f"Tensor ID: {batch.ids[i]}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()