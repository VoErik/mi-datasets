import pytest
import os
import shutil

@pytest.fixture(autouse=True)
def mock_cache_env(monkeypatch, tmp_path):
    """Forces all tests to use a temporary cache directory."""
    test_cache = tmp_path / "mi_test_cache"
    monkeypatch.setenv("MI_DATASETS_CACHE", str(test_cache))
    yield
    if test_cache.exists():
        shutil.rmtree(test_cache)