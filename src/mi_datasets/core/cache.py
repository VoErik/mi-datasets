import os
import shutil
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

class CacheManager:
    """Centralized I/O and cache management for MI Datasets."""
    
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = os.path.expanduser(
            base_dir or os.environ.get("MI_DATASETS_CACHE", "~/.cache/mi_datasets")
        )
        os.makedirs(self.base_dir, exist_ok=True)

    def get_dataset_dir(self, identifier: str) -> str:
        """
        Returns the isolated cache directory for a specific dataset.
        Maintains the hierarchy (e.g., ~/.cache/mi_datasets/vision/cifar10)
        """
        path = os.path.join(self.base_dir, *identifier.split("/"))
        os.makedirs(path, exist_ok=True)
        return path

    def is_cached(self, path: str, required_files: Optional[List[str]] = None) -> bool:
        """Verifies if a dataset exists and is completely downloaded."""
        if not os.path.exists(path):
            return False
            
        if required_files:
            return all(os.path.exists(os.path.join(path, f)) for f in required_files)
            
        return len(os.listdir(path)) > 0

    def clear_cache(self, identifier: Optional[str] = None) -> None:
        """Purges the cache. If identifier is None, nukes the entire root cache."""
        target_dir = self.get_dataset_dir(identifier) if identifier else self.base_dir
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
            logger.info(f"Cleared cache at: {target_dir}")
            
        if not identifier:
            os.makedirs(self.base_dir, exist_ok=True)