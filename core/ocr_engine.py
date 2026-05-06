"""
OCR 引擎模块（基于 EasyOCR）
支持多区域识别与帧级哈希缓存
"""

import numpy as np
import easyocr
from typing import List, Dict, Optional
from core.cache_manager import CacheManager


class OCRResult:
    """OCR 识别结果对象"""
    def __init__(self, text: str, confidence: float, bbox: List[List[int]]):
        self.text = text
        self.confidence = confidence
        self.bbox = bbox

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bbox": self.bbox
        }


class OCREngine:
    def __init__(
        self,
        languages: List[str] = ['ch_sim', 'en'],
        use_gpu: bool = True,
        cache_dir: str = "temp/cache/ocr"
    ):
        self.languages = languages
        self.use_gpu = use_gpu
        self.cache = CacheManager(cache_dir)

        # 初始化 EasyOCR reader
        self.reader = easyocr.Reader(
            lang_list=languages,
            gpu=use_gpu
        )

        # 多区域识别策略（可扩展）
        self.region_strategies = [
            self._full_frame_strategy,
        ]

    def _full_frame_strategy(self, frame: np.ndarray) -> List[OCRResult]:
        """全图识别策略"""
        raw_results = self.reader.readtext(frame, detail=1)
        results = []
        for bbox, text, confidence in raw_results:
            if confidence > 0.2:  # 过滤低置信度结果
                results.append(OCRResult(
                    text=text,
                    confidence=confidence,
                    bbox=[[int(pt[0]), int(pt[1])] for pt in bbox]
                ))
        return results

    def recognize(self, frame: np.ndarray) -> List[OCRResult]:
        """
        识别单帧图像中的文字
        Args:
            frame: BGR 图像 (H, W, 3)
        Returns:
            OCRResult 列表
        """
        # 计算输入哈希
        frame_bytes = frame.tobytes()
        cache_key = CacheManager.compute_hash(frame_bytes)

        # 尝试加载缓存
        cached = self.cache.load(cache_key)
        if cached is not None:
            return cached

        # 执行多策略识别（选取第一个有结果的）
        results = []
        for strategy in self.region_strategies:
            strategy_results = strategy(frame)
            if strategy_results:
                results.extend(strategy_results)
                break

        # 保存到缓存
        self.cache.save(cache_key, results)
        return results

    def recognize_batch(
        self,
        frames: List[np.ndarray],
        batch_size: int = 16
    ) -> List[List[OCRResult]]:
        """
        批量识别多帧
        Args:
            frames: 帧列表
            batch_size: 批处理大小（暂用于逐帧处理，可后续优化）
        Returns:
            各帧的识别结果列表
        """
        all_results = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batch_results = [self.recognize(frame) for frame in batch]
            all_results.extend(batch_results)
        return all_results

    def _postprocess(self, results: List[OCRResult]) -> List[OCRResult]:
        """后处理（去重、排序等）"""
        # 简单按置信度降序
        return sorted(results, key=lambda x: x.confidence, reverse=True)