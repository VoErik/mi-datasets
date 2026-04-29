import os
from typing import Any, Dict
from datasets import load_dataset as hf_load_dataset
from mi_datasets.core.base import BaseMIDataset
from mi_datasets.core.registry import register_dataset

from dotenv import load_dotenv
load_dotenv("/Users/erik/Desktop/PROJECTS/research-harness/mi_datasets/.env")

@register_dataset("huggingface")
class HuggingFaceDataset(BaseMIDataset):
    def _download(self) -> None:
        pass # Handled by hf_load_dataset

    def _load_into_memory(self) -> None:
        if "path" not in self.config:
            raise ValueError("HuggingFace provider requires a 'path' in the config (e.g., 'mnist').")

        hf_token = os.getenv("HF_TOKEN", None)

        self.hf_dataset = hf_load_dataset(
            path=self.config["path"],
            name=self.config.get("name"),
            split=self.config.get("split", "train"),
            cache_dir=self.cache_dir,
            trust_remote_code=self.config.get("trust_remote_code", False),
            token=hf_token
        )
        
        self.input_col = self.config.get("input_col", "image")
        self.target_col = self.config.get("target_col", "label")

    def _get_full_length(self) -> int:
        return len(self.hf_dataset)

    def _get_raw_data(self, idx: int) -> Dict[str, Any]:
        row = self.hf_dataset[idx]
        
        if self.input_col not in row:
            raise KeyError(f"Input column '{self.input_col}' not found in HuggingFace row keys: {list(row.keys())}")
            
        raw_input = row.pop(self.input_col)
        target = row.pop(self.target_col) if self.target_col in row else None
        metadata = {"hf_metadata": row}

        return {
            "id": idx,
            "input": raw_input,
            "target": target,
            "metadata": metadata
        }