import os
import yaml
from pathlib import Path
from typing import Type, Callable, Dict, Any, Optional, Union
from mi_datasets.core.base import BaseMIDataset

_DATASET_REGISTRY: Dict[str, Type[BaseMIDataset]] = {}

def list_available_datasets() -> list[str]:
    """
    Returns a sorted list of all registered dataset identifiers.
    """
    return sorted(list(_DATASET_REGISTRY.keys()))

def register_dataset(name: str) -> Callable:
    """
    Decorator to register a dataset class under a specific string identifier.
    """
    def wrapper(cls: Type[BaseMIDataset]) -> Type[BaseMIDataset]:
        if name in _DATASET_REGISTRY:
            raise KeyError(f"Dataset identifier '{name}' is already registered to {_DATASET_REGISTRY[name].__name__}.")
        if not issubclass(cls, BaseMIDataset):
            raise TypeError(f"Class '{cls.__name__}' must inherit from BaseMIDataset to be registered.")
        
        _DATASET_REGISTRY[name] = cls
        return cls
    return wrapper

def load_dataset(
    identifier: Union[str, Path], 
    config: Optional[Dict[str, Any]] = None, 
    **kwargs
) -> BaseMIDataset:
    """
    Factory method to instantiate a dataset by its string identifier OR a YAML config file.
    """
    config = config or {}
    identifier_str = str(identifier)
    
    if identifier_str.endswith((".yaml", ".yml")):
        if not os.path.exists(identifier_str):
            raise FileNotFoundError(f"Config file not found: {identifier_str}")
        
        with open(identifier_str, "r") as f:
            yaml_data = yaml.safe_load(f) or {}
            
        if "dataset" not in yaml_data:
            raise ValueError(f"YAML config '{identifier_str}' must contain a 'dataset' key (e.g., dataset: 'vision/cifar10').")
        
        target_identifier = yaml_data.pop("dataset")
        
        yaml_data.update(config)
        yaml_data.update(kwargs)
        
        return load_dataset(target_identifier, config=yaml_data)

    if identifier not in _DATASET_REGISTRY:
        available = list(_DATASET_REGISTRY.keys())
        raise KeyError(
            f"Dataset '{identifier}' not found in registry. "
            f"Available datasets: {available}. "
            f"Ensure the provider module has been imported."
        )
    
    default_cache = os.environ.get("MI_DATASETS_CACHE", "~/.cache/mi_datasets")
    cache_dir = kwargs.pop("cache_dir", default_cache)
    transform = kwargs.pop("transform", None)
    target_transform = kwargs.pop("target_transform", None)
    
    config.update(kwargs)
    
    config["_identifier"] = str(identifier)
    
    dataset_cls = _DATASET_REGISTRY[str(identifier)]
    
    return dataset_cls(
        config=config, 
        cache_dir=cache_dir,
        transform=transform,
        target_transform=target_transform
    )