import os
from abc import ABC, abstractmethod
from typing import (
    Any, 
    Callable, 
    Dict, 
    List, 
    Optional
)
from torch.utils.data import Dataset, DataLoader
from mi_datasets.core.type import DataItem, DataBatch, DatasetInfo, TransformMeta
from mi_datasets.core.cache import CacheManager
import numpy as np

class BaseMIDataset(Dataset, ABC):
    """
    Template Base Class for all Mechanistic Interpretability Datasets.
    Enforces caching, transforms, and strict return schemas.
    """
    _modality: str = "unknown"

    def __init__(
        self, 
        config: dict, 
        cache_dir: str = "~/.cache/mi_datasets",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        self.config = config
        self.split = config.get("split", "train")

        self.keep_raw_input = config.get("keep_raw_input", True)
        self.keep_metadata = config.get("keep_metadata", True)
        
        self.transform = transform
        self.target_transform = target_transform
        self.cache = CacheManager(base_dir=cache_dir)
        
        identifier = self.config.get("_identifier", "unknown/unknown")
        self.cache_dir = self.cache.get_dataset_dir(identifier)
        
        self.data = None
        self._setup()

    def _setup(self) -> None:
        """The Template Method controlling initialization."""
        if not self._is_cached():
            self._download()
        self._load_into_memory()
        
        self.total_items = self._get_full_length()
        self._indices = self._generate_subset_indices()

    def _generate_subset_indices(self) -> list[int]:
        """Creates a deterministic subset of indices if subset_fraction < 1.0"""
        fraction = self.config.get("subset_fraction", 1.0)
        seed = self.config.get("seed", 42)
        
        if fraction >= 1.0:
            return list(range(self.total_items))
            
        num_samples = max(1, int(self.total_items * fraction))
        
        rng = np.random.default_rng(seed)
        indices = rng.choice(self.total_items, size=num_samples, replace=False).tolist()
        indices.sort()
        return indices

    def _is_cached(self, required_files: Optional[List[str]] = None) -> bool:
        """Check if the dataset exists locally via the Cache Manager."""
        return self.cache.is_cached(self.cache_dir, required_files)

    @abstractmethod
    def _download(self) -> None:
        """Fetch the dataset from the external provider."""
        pass

    @abstractmethod
    def _load_into_memory(self) -> None:
        """Load data from disk to self.data (e.g., memory-mapping or RAM)."""
        pass

    @abstractmethod
    def _get_full_length(self) -> int:
        """Subclasses MUST implement this instead of __len__"""
        pass

    def __len__(self) -> int:
        """Returns the length of the deterministically sampled subset."""
        return len(self._indices)

    @abstractmethod
    def _get_raw_data(self, idx: int) -> Dict[str, Any]:
        """
        Subclasses MUST implement this to fetch the raw data.
        Expected return format: {"id": ..., "input": ..., "target": ..., "metadata": ...}
        """
        pass

    def __getitem__(self, idx: int) -> DataItem:
        actual_idx = self._indices[idx]
        raw_dict = self._get_raw_data(actual_idx)
        
        raw_input = raw_dict["input"]
        raw_target = raw_dict.get("target")
        metadata = raw_dict.get("metadata", {})

        if self.transform:
            result = self.transform(raw_input)
            if isinstance(result, tuple) and len(result) == 2:
                model_input, transform_history = result
                metadata["transform_history"] = transform_history
            else:
                model_input = result
        else:
            model_input = raw_input

        processed_target = self.target_transform(raw_target) if self.target_transform and raw_target is not None else raw_target

        return DataItem(
            id=raw_dict.get("id", actual_idx),
            model_input=model_input,
            raw_input=raw_input if self.keep_raw_input else None,
            targets=processed_target,
            metadata=metadata if self.keep_metadata else None
        )

    def get_collate_fn(self) -> Callable[[list[DataItem]], DataBatch]:
        """
        Returns the collator. Base class provides a standard PyTorch stacker.
        Graph subclasses override this to return PyGBatch collators.
        """
        def default_collate(batch: list[DataItem]) -> DataBatch:
            import torch
            return DataBatch(
                ids=[item.id for item in batch],
                model_inputs=torch.stack([item.model_input for item in batch]),
                raw_inputs=[item.raw_input for item in batch],
                targets=torch.stack([item.targets for item in batch]) if batch[0].targets is not None else None,
                metadata=[item.metadata for item in batch]
            )
        return default_collate

    def get_dataloader(self, batch_size: int, shuffle: bool = False, num_workers: int = 0) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.get_collate_fn()
        )
    
    def _parse_transforms(self, transform_callable: Optional[Callable]) -> List[TransformMeta]:
        """
        Attempts to extract serialization metadata from the transform pipeline.
        """
        if transform_callable is None:
            return []

        parsed_transforms = []
        
        if hasattr(transform_callable, "transforms"): 
            components = transform_callable.transforms
        else:
            components = [transform_callable]

        for t in components:
            if hasattr(t, "to_dict"):
                assert hasattr(t, "to_dict"), "Transform must be serializable."
                params = t.to_dict()
            else:
                # Fallback: inspect the object dictionary, filtering out internal/callable attributes
                params = {k: v for k, v in t.__dict__.items() if not k.startswith("_") and not callable(v)}
            
            parsed_transforms.append(TransformMeta(
                name=t.__class__.__name__,
                params=params,
                is_spatial=any(kw in t.__class__.__name__.lower() for kw in ["crop", "resize", "pad", "drop", "pool"]) # TODO: maintain list of spatial transforms
            ))
            
        return parsed_transforms

    def get_info(self) -> DatasetInfo:
        safe_idx = self._indices[0] if self._indices else 0
        sample = self._get_raw_data(safe_idx)
        feature_types = {k: type(v).__name__ for k, v in sample.items()}

        modality_metadata = self._get_modality_metadata()

        identifier = self.config.get("_identifier", "unknown")
        name = identifier.split("/")[-1] 

        return DatasetInfo(
            name=name,
            modality=self._modality,
            provider=self.__class__.__name__,
            split=self.split,
            num_items=len(self),
            features=feature_types,
            transforms=self._parse_transforms(self.transform),
            target_transforms=self._parse_transforms(self.target_transform),
            metadata=modality_metadata
        )

    def _get_modality_metadata(self) -> Dict[str, Any]:
        """Subclasses override this to add specific info (e.g., num_classes, num_node_features)."""
        return {}