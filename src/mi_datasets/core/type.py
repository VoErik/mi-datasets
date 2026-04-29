from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    Any, 
    Dict,
    List,
    Optional, 
    Union, 
    TYPE_CHECKING
)
import torch

if TYPE_CHECKING:
    from torch_geometric.data import Data as PyGData
    from torch_geometric.data import Batch as PyGBatch

@dataclass
class DataItem:
    """
    A single unbatched item from any dataset.
    """
    id: Union[str, int]
    model_input: Union[torch.Tensor, "PyGData"]
    raw_input: Any
    targets: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class DataBatch:
    ids: list[Union[str, int]]
    model_inputs: Union[torch.Tensor, "PyGBatch"]
    raw_inputs: list[Any]
    targets: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None
    metadata: Optional[list[Dict[str, Any]]] = None

    def __iter__(self):
        """Allows standard PyTorch tuple unpacking: inputs, targets = batch"""
        yield self.model_inputs
        yield self.targets

    def __getitem__(self, idx: int):
        """Allows standard PyTorch indexing: batch[0] -> inputs, batch[1] -> targets"""
        if idx == 0:
            return self.model_inputs
        elif idx == 1:
            return self.targets
        else:
            raise IndexError("DataBatch only supports indexing 0 (inputs) and 1 (targets) for PyTorch compatibility.")

    def to(self, device: torch.device) -> "DataBatch":
        """Recursively moves tensors to the target device."""
        model_inputs = self.model_inputs.to(device) if hasattr(self.model_inputs, 'to') else self.model_inputs
        
        targets = self.targets
        if isinstance(targets, torch.Tensor):
            targets = targets.to(device)
        elif isinstance(targets, dict):
            targets = {k: v.to(device) for k, v in targets.items()}

        return DataBatch(
            ids=self.ids,
            model_inputs=model_inputs,
            raw_inputs=self.raw_inputs,
            targets=targets,
            metadata=self.metadata
        )

@dataclass
class TransformMeta:
    name: str
    params: Dict[str, Any]
    is_spatial: bool

@dataclass
class DatasetInfo:
    name: str
    modality: str
    provider: str
    split: str
    num_items: int
    features: Dict[str, str]
    transforms: List[TransformMeta]
    target_transforms: List[TransformMeta]
    metadata: Dict[str, Any]

    def __str__(self) -> str:
        lines = [
            f"Dataset: {self.name.upper()} [{self.modality}]",
            f"Provider: {self.provider}",
            f"Split: {self.split} | Total Items: {self.num_items}",
            "-" * 30,
            "Features:"
        ]
        
        for k, v in self.features.items():
            lines.append(f"  • {k}: {v}")

        if self.transforms:
            lines.append("\nInput Transforms:")
            for t in self.transforms:
                lines.append(f"  • {t.name} (Spatial: {t.is_spatial})")
                for pk, pv in t.params.items():
                    lines.append(f"      {pk}: {pv}")

        if self.metadata:
            lines.append("\nMetadata:")
            for k, v in self.metadata.items():
                if isinstance(v, list) and len(v) > 5:
                    v_str = f"[{v[0]}, {v[1]}, ..., {v[-1]}] (len={len(v)})"
                else:
                    v_str = str(v)
                lines.append(f"  • {k}: {v_str}")

        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return self.__str__()