import os
import torch
import torchvision
import mi_datasets.providers.torchvision_datasets
from typing import Any, Dict, Optional, List
from pathlib import Path
from mi_datasets.core.base import BaseMIDataset

from mi_datasets.core.registry import register_dataset

class _CoreCIFARDataset(BaseMIDataset):
    _modality = "vision"
    _cifar_version = None 

    def _get_tv_class(self):
        if self._cifar_version == "cifar10":
            return torchvision.datasets.CIFAR10
        elif self._cifar_version == "cifar100":
            return torchvision.datasets.CIFAR100
        else:
            raise ValueError("Cifar Version must either be 'cifar10' or 'cifar100'")

    def _is_cached(self, required_files: Optional[List[str]] = None) -> bool:
        """Overrides base cache check to enforce specific Torchvision extracted folders."""
        folder_name = "cifar-10-batches-py" if self._cifar_version == "cifar10" else "cifar-100-python"
        return super()._is_cached(required_files=[folder_name])

    def _download(self) -> None:
        pass

    def _load_into_memory(self) -> None:
        dataset_cls = self._get_tv_class()
        is_train = self.config.get("split", "train") == "train"
        
        self.tv_dataset = dataset_cls(
            root=self.cache_dir, 
            train=is_train, 
            download=not self._is_cached()
        )

    def _get_full_length(self) -> int:
        return len(self.tv_dataset)

    def _get_raw_data(self, idx: int) -> Dict[str, Any]:
        img, target = self.tv_dataset[idx]
        class_name = self.tv_dataset.classes[target]
        
        return {
            "id": idx,
            "input": img,
            "target": torch.tensor(target, dtype=torch.long),
            "metadata": {
                "class_name": class_name
            }
        }

    def _get_modality_metadata(self) -> Dict[str, Any]:
        return {
            "classes": self.tv_dataset.classes,
            "num_classes": len(self.tv_dataset.classes)
        }

@register_dataset("torchvision/cifar10")
class CIFAR10Dataset(_CoreCIFARDataset):
    _modality = "vision"
    _cifar_version = "cifar10"

@register_dataset("torchvision/cifar100")
class CIFAR100Dataset(_CoreCIFARDataset):
    _modality = "vision"
    _cifar_version = "cifar100"
    
@register_dataset("torchvision/celeba")
class CelebADataset(BaseMIDataset):
    _modality = "vision"
    
    def _is_cached(self, required_files: Optional[List[str]] = None) -> bool:
        """Overrides base cache check to enforce the CelebA extracted folder."""
        return super()._is_cached(required_files=["celeba"])

    def _download(self) -> None:
        pass # Handled by torchvision download=True

    def _load_into_memory(self) -> None:
        self.tv_dataset = torchvision.datasets.CelebA(
            root=str(Path(self.cache_dir).parent), # torchvision celeba is quirky like that :)
            split=self.split,
            target_type="attr",
            download=not self._is_cached()
        )

    def _get_full_length(self) -> int:
        return len(self.tv_dataset)

    def _get_raw_data(self, idx: int) -> Dict[str, Any]:
        img, attr = self.tv_dataset[idx]
        
        return {
            "id": idx,
            "input": img,
            "target": attr.clone().detach(), # 40-dim multi-hot tensor
            "metadata": {} 
        }

    def _get_modality_metadata(self) -> Dict[str, Any]:
        return {
            "num_attributes": 40,
            "attribute_names": self.tv_dataset.attr_names
        }
