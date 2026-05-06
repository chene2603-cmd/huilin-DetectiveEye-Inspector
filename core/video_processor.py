"""
视频处理模块
负责场景检测、关键帧提取与缓存管理
"""

import os
import json
import hashlib
import pickle
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime


class VideoProcessor:
    def __init__(
        self,
        preview_fps: float = 1.0,
        detail_fps: float = 6.0,
        scene_threshold: float = 0.3,
        cache_dir: str = "temp/cache"
    ):
        """
        Args:
            preview_fps: 快速预览模式的采样帧率（每秒提取多少帧）
            detail_fps: 精细分析模式的采样帧率
            scene_threshold: 场景切换检测阈值（0-1），直方图差异超过此值视为新场景
            cache_dir: 缓存目录路径
        """
        self.preview_fps = preview_fps
        self.detail_fps = detail_fps
        self.scene_threshold = scene_threshold
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_video_info(self, video_path: str) -> Dict:
        """获取视频基本信息"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
            if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
        }
        cap.release()
        return info

    def _compute_hash(self, data: bytes) -> str:
        """计算数据哈希值"""
        return hashlib.md5(data).hexdigest()

    def _get_cache_path(self, video_path: str) -> Path:
        """根据视频路径生成缓存文件路径"""
        # 使用视频路径的哈希作为缓存文件名
        path_hash = self._compute_hash(video_path.encode("utf-8"))
        return self.cache_dir / f"{path_hash}.pkl"

    def _load_cache(self, video_path: str) -> Optional[List[Dict]]:
        """尝试从缓存加载帧信息"""
        cache_path = self._get_cache_path(video_path)
        if not cache_path.exists():
            return None

        # 检查视频文件是否被修改（通过文件哈希）
        try:
            with open(video_path, "rb") as f:
                current_hash = self._compute_hash(f.read())
        except Exception:
            return None

        try:
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)
            if cache_data.get("video_hash") == current_hash:
                print(f"🔄 从缓存加载 {len(cache_data['frames'])} 个关键帧")
                return cache_data["frames"]
        except Exception as e:
            print(f"⚠️  缓存损坏，重新处理: {e}")
        return None

    def _save_cache(self, video_path: str, frames: List[Dict]):
        """保存帧信息到缓存"""
        cache_path = self._get_cache_path(video_path)
        try:
            # 计算当前视频文件哈希
            with open(video_path, "rb") as f:
                video_hash = self._compute_hash(f.read())

            cache_data = {
                "video_hash": video_hash,
                "frames": frames,
                "timestamp": datetime.now().isoformat()
            }
            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f)
            print(f"💾 缓存已保存：{len(frames)} 个关键帧")
        except Exception as e:
            print(f"⚠️  缓存保存失败: {e}")

    @staticmethod
    def _calculate_histogram_diff(hist1: np.ndarray, hist2: np.ndarray) -> float:
        """计算两个直方图的差异（巴氏距离）"""
        # 归一化直方图
        hist1 = hist1 / (np.sum(hist1) + 1e-7)
        hist2 = hist2 / (np.sum(hist2) + 1e-7)
        # 巴氏系数
        bc = np.sum(np.sqrt(hist1 * hist2))
        return 1.0 - bc

    def _detect_scene_changes(
        self,
        frames: List[np.ndarray],
        timestamps: List[float],
        frame_indices: List[int],
        base_histograms: Optional[List[np.ndarray]] = None
    ) -> List[int]:
        """
        基于直方图差异检测场景切换位置
        返回场景起始帧的索引列表（至少包含第一帧）
        """
        if len(frames) <= 1:
            return [0]

        # 计算每帧的直方图（HSV空间，亮度通道）
        histograms = []
        for frame in frames:
            if frame is None:
                histograms.append(np.zeros(256))
                continue
            # 转HSV并计算H通道直方图
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            histograms.append(hist)

        # 找出差异超过阈值的点
        scene_starts = [0]  # 第一帧总是场景开始
        for i in range(1, len(histograms)):
            diff = self._calculate_histogram_diff(histograms[i-1], histograms[i])
            if diff > self.scene_threshold:
                scene_starts.append(i)

        # 避免场景片段过短（至少间隔5帧）
        min_gap = 5
        filtered = [0]
        for start in scene_starts[1:]:
            if start - filtered[-1] >= min_gap:
                filtered.append(start)
        scene_starts = filtered

        return scene_starts

    def _extract_keyframes(
        self,
        video_path: str,
        target_fps: float,
        use_scene_detection: bool = True
    ) -> Tuple[List[np.ndarray], List[float], List[int]]:
        """
        从视频中提取关键帧
        Returns:
            frames: 帧图像列表
            timestamps: 对应的时间戳（秒）
            frame_indices: 原始帧索引
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if original_fps <= 0:
            original_fps = 30.0  # 默认假设

        # 确定采样间隔
        if target_fps <= 0:
            target_fps = original_fps
        step = max(1, int(original_fps / target_fps))

        frames = []
        timestamps = []
        frame_indices = []

        print(f"📹 开始提取关键帧 (原始FPS: {original_fps:.1f}, 目标FPS: {target_fps})")
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if count % step == 0:
                frames.append(frame)
                timestamps.append(count / original_fps)
                frame_indices.append(count)
            count += 1

        cap.release()
        print(f"✅ 共提取 {len(frames)} 个候选帧")

        # 如果启用场景检测，进一步筛选
        if use_scene_detection and len(frames) > 10:
            scene_starts = self._detect_scene_changes(frames, timestamps, frame_indices)

            # 每个场景保留第一帧及其后的一帧（可调整）
            scene_frames = []
            scene_timestamps = []
            scene_indices = []

            for i, start_idx in enumerate(scene_starts):
                # 添加场景第一帧
                scene_frames.append(frames[start_idx])
                scene_timestamps.append(timestamps[start_idx])
                scene_indices.append(frame_indices[start_idx])

                # 添加场景内的中间帧（如果场景长度足够）
                end_idx = scene_starts[i+1] if i+1 < len(scene_starts) else len(frames)
                if end_idx - start_idx > 5:
                    mid = start_idx + (end_idx - start_idx) // 2
                    scene_frames.append(frames[mid])
                    scene_timestamps.append(timestamps[mid])
                    scene_indices.append(frame_indices[mid])

            return scene_frames, scene_timestamps, scene_indices

        return frames, timestamps, frame_indices

    def extract_frames(
        self,
        video_path: str,
        use_cache: bool = True,
        fast_mode: bool = False
    ) -> List[Dict]:
        """
        提取视频关键帧的主接口

        Args:
            video_path: 视频文件路径
            use_cache: 是否使用缓存
            fast_mode: 快速模式，使用较低的采样率和简化的场景检测

        Returns:
            List[Dict]: 每个元素包含 {
                "frame": np.ndarray (HxWxC BGR),
                "timestamp": float (秒),
                "frame_index": int (原始帧号),
                "scene_id": int (场景编号，从0开始)
            }
        """
        # 尝试加载缓存
        if use_cache:
            cached = self._load_cache(video_path)
            if cached is not None:
                return cached

        # 确定采样参数
        target_fps = self.preview_fps if fast_mode else self.detail_fps
        use_scene = not fast_mode  # 快速模式关闭场景检测以提速

        try:
            frames, timestamps, indices = self._extract_keyframes(
                video_path,
                target_fps=target_fps,
                use_scene_detection=use_scene
            )
        except Exception as e:
            raise RuntimeError(f"提取关键帧失败: {e}")

        # 构建结果
        result = []
        scene_id = 0
        prev_scene = -1

        # 如果需要场景ID，可以基于场景检测结果分配
        # 这里简单根据帧索引间的大幅跳跃来标记场景切换
        for i, (frame, ts, idx) in enumerate(zip(frames, timestamps, indices)):
            if i > 0:
                # 如果帧索引跳跃超过10帧，视作新场景
                if idx - indices[i-1] > 10:
                    scene_id += 1
            elif i == 0:
                scene_id = 0

            result.append({
                "frame": frame,
                "timestamp": ts,
                "frame_index": idx,
                "scene_id": scene_id
            })

        # 保存缓存
        if use_cache:
            self._save_cache(video_path, result)

        return result

    def extract_frame_at_time(
        self,
        video_path: str,
        timestamp: float
    ) -> Optional[Dict]:
        """提取指定时间点附近的帧（用于证据回溯）"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        target_frame_idx = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        return {
            "frame": frame,
            "timestamp": timestamp,
            "frame_index": target_frame_idx,
            "scene_id": -1  # 单独提取的帧不标场景
        }

    def get_video_duration(self, video_path: str) -> float:
        """获取视频时长（秒）"""
        info = self._get_video_info(video_path)
        return info["duration"]