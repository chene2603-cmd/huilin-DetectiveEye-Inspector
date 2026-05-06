# core/detector.py
"""
目标检测模块（基于 YOLOv8）
支持 GPU/CPU 自适应与帧级哈希缓存
"""

import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
from typing import List, Dict, Optional
from core.cache_manager import CacheManager


class YOLODetector:
    def __init__(
        self,
        model_path: str = "models/yolov8n.pt",
        confidence: float = 0.4,
        device: str = "auto",
        cache_dir: str = "temp/cache/yolo"
    ):
        self.confidence = confidence
        self.device = self._resolve_device(device)
        self.cache = CacheManager(cache_dir)

        # 加载模型
        if not Path(model_path).exists():
            print(f"⬇️  模型未找到，正在下载 {model_path} ...")
        self.model = YOLO(model_path)
        self.model.to(self.device)

        # 预热模型（避免首次推理延迟）
        self._warmup()

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                print("🔍 检测到 CUDA，使用 GPU 推理")
                return "cuda"
            print("🖥️  未检测到 CUDA，使用 CPU 推理")
            return "cpu"
        return device

    def _warmup(self):
        """用空白图像预热模型"""
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        try:
            _ = self.model(dummy, verbose=False)
        except Exception:
            pass

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        单帧检测
        Args:
            frame: BGR 图像 (H, W, 3)
        Returns:
            检测结果列表，每个元素包含 {label, confidence, bbox, ...}
        """
        # 计算输入哈希
        frame_bytes = frame.tobytes()
        cache_key = CacheManager.compute_hash(frame_bytes)

        # 尝试加载缓存
        cached = self.cache.load(cache_key)
        if cached is not None:
            return cached

        # 执行推理
        results = self.model(frame, conf=self.confidence, verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                detections.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": xyxy,          # [x1, y1, x2, y2]
                    "class_id": cls_id
                })

        # 写入缓存
        self.cache.save(cache_key, detections)
        return detections

    def detect_batch(
        self,
        frames: List[np.ndarray],
        batch_size: int = 8
    ) -> List[List[Dict]]:
        """
        批量检测
        Args:
            frames: 帧图像列表
            batch_size: 批处理大小
        Returns:
            与 frames 等长的列表，每个元素是该帧的检测结果
        """
        all_detections = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batch_results = [self.detect(frame) for frame in batch]
            all_detections.extend(batch_results)
        return all_detections