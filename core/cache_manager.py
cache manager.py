"""
通用缓存管理器
基于输入数据的哈希值，自动缓存和加载 Python 对象（pickle）
"""

import hashlib
import pickle
from pathlib import Path
from typing import Any, Optional, Callable


class CacheManager:
    def __init__(self, cache_dir: str = "temp/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_hash(data: bytes) -> str:
        return hashlib.md5(data).hexdigest()

    def _get_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pkl"

    def load(self, key: str) -> Optional[Any]:
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            cache_path.unlink(missing_ok=True)
            return None

    def save(self, key: str, data: Any):
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"⚠️  缓存保存失败: {e}")

    def get_or_compute(
        self, key: str, compute_func: Callable[[], Any],
        force_refresh: bool = False
    ) -> Any:
        if not force_refresh:
            cached = self.load(key)
            if cached is not None:
                return cached
        result = compute_func()
        self.save(key, result)
        return result

    def invalidate(self, key: str):
        cache_path = self._get_cache_path(key)
        cache_path.unlink(missing_ok=True)

    def clear_all(self):
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        print("🧹 缓存已全部清空")，