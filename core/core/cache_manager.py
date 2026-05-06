# core/cache_manager.py
"""
通用缓存管理器
基于输入数据的哈希值，自动缓存和加载 Python 对象（pickle）
"""

import os
import hashlib
import pickle
from pathlib import Path
from typing import Any, Optional, Callable


class CacheManager:
    """缓存管理器，用于缓存耗时函数的计算结果"""

    def __init__(self, cache_dir: str = "temp/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_hash(data: bytes) -> str:
        """计算数据的 MD5 哈希"""
        return hashlib.md5(data).hexdigest()

    def _get_cache_path(self, key: str) -> Path:
        """根据缓存键生成文件路径"""
        return self.cache_dir / f"{key}.pkl"

    def load(self, key: str) -> Optional[Any]:
        """
        尝试从缓存加载数据
        Args:
            key: 缓存键（通常是基于输入数据计算的哈希）
        Returns:
            缓存的数据，若不存在或损坏则返回 None
        """
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception:
            # 缓存损坏，删除文件
            cache_path.unlink(missing_ok=True)
            return None

    def save(self, key: str, data: Any):
        """将数据保存到缓存"""
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"⚠️  缓存保存失败: {e}")

    def get_or_compute(
        self,
        key: str,
        compute_func: Callable[[], Any],
        force_refresh: bool = False
    ) -> Any:
        """
        缓存穿透模式：有缓存直接返回，否则调用 compute_func 计算并缓存
        Args:
            key: 缓存键
            compute_func: 计算的函数（无参数）
            force_refresh: 是否强制重新计算
        Returns:
            计算或缓存的结果
        """
        if not force_refresh:
            cached = self.load(key)
            if cached is not None:
                return cached
        result = compute_func()
        self.save(key, result)
        return result

    def invalidate(self, key: str):
        """删除指定缓存"""
        cache_path = self._get_cache_path(key)
        cache_path.unlink(missing_ok=True)

    def clear_all(self):
        """清空所有缓存"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        print("🧹 缓存已全部清空")
