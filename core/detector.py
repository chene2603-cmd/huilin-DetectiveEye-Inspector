"""
视觉感知层 - YOLO目标检测
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
from dataclasses import dataclass
from ultralytics import YOLO
import threading

@dataclass
class Detection:
    """检测结果"""
    label: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    frame_index: int
    timestamp: float


class YOLODetector:
    """YOLO检测器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_config = config.get('models', {}).get('yolo', {})
        
        # 模型路径
        model_path = self.model_config.get(
            'model_path', 
            'models/yolov8n.pt'
        )
        
        # 设备选择
        self.device = self._select_device()
        print(f"🖥️  使用设备: {self.device}")
        
        # 加载模型
        print(f"🔧 加载YOLO模型: {model_path}")
        self.model = self._load_model(model_path)
        
        # 检测类别
        self.classes = self._get_classes()
        
        # 批次处理
        self.batch_size = self.model_config.get('batch_size', 8)
        self.confidence_threshold = self.model_config.get('confidence', 0.4)
        
        # 缓存
        self.cache = {}
        self.cache_lock = threading.Lock()
    
    def _select_device(self):
        """选择设备（GPU/CPU）"""
        if torch.cuda.is_available():
            gpu_memory_limit = self.model_config.get('gpu_memory_limit', 0.8)
            
            # 检查显存
            torch.cuda.empty_cache()
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated(0)
            
            if allocated / total_memory < (1 - gpu_memory_limit):
                return 'cuda'
        
        return 'cpu'
    
    def _load_model(self, model_path: str):
        """加载YOLO模型"""
        # 如果模型不存在，尝试下载
        if not Path(model_path).exists():
            print(f"⚠️  模型文件不存在: {model_path}")
            print("正在下载预训练模型...")
            model = YOLO('yolov8n.pt')
            model.save(model_path)
            print(f"✅ 模型已保存到: {model_path}")
        else:
            model = YOLO(model_path)
        
        # 设置为推理模式
        model.to(self.device)
        model.eval()
        
        return model
    
    def _get_classes(self):
        """获取检测类别"""
        # 默认类别
        default_classes = [
            'person', 'cell phone', 'laptop', 'book',
            'bottle', 'cup', 'fire', 'smoke', 'knife'
        ]
        
        # 从配置读取自定义类别
        custom_classes = self.model_config.get('custom_classes', [])
        
        return list(set(default_classes + custom_classes))
    
    def warmup(self, warmup_iters: int = 10):
        """预热模型"""
        print("🔥 预热模型...")
        
        dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
        
        with torch.no_grad():
            for _ in range(warmup_iters):
                _ = self.model(dummy_input)
        
        print("✅ 模型预热完成")
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """单帧检测"""
        # 检查缓存
        cache_key = hash(frame.data.tobytes())
        with self.cache_lock:
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        # 预处理
        input_tensor = self._preprocess(frame)
        
        # 推理
        with torch.no_grad():
            results = self.model(input_tensor, verbose=False)
        
        # 后处理
        detections = self._postprocess(results, frame)
        
        # 缓存结果
        with self.cache_lock:
            self.cache[cache_key] = detections
        
        return detections
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """批次检测"""
        all_detections = []
        
        # 分批处理
        for i in range(0, len(frames), self.batch_size):
            batch_frames = frames[i:i+self.batch_size]
            
            # 预处理批次
            batch_tensors = [self._preprocess(frame) for frame in batch_frames]
            
            # 堆叠批次
            if len(batch_tensors) > 1:
                batch = torch.cat(batch_tensors, dim=0)
            else:
                batch = batch_tensors[0]
            
            # 推理
            with torch.no_grad():
                results = self.model(batch, verbose=False)
            
            # 后处理
            for j, result in enumerate(results):
                if hasattr(result, 'boxes'):
                    frame_idx = i + j
                    frame = batch_frames[j]
                    detections = self._postprocess(result, frame, frame_idx)
                    all_detections.append(detections)
                else:
                    all_detections.append([])
        
        return all_detections
    
    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """预处理帧"""
        # 调整大小
        resized = cv2.resize(frame, (640, 640))
        
        # 转换通道顺序 BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 归一化
        normalized = rgb.astype(np.float32) / 255.0
        
        # 转换维度 HWC -> CHW
        chw = normalized.transpose(2, 0, 1)
        
        # 转换为Tensor并添加批次维度
        tensor = torch.from_numpy(chw).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def _postprocess(self, result, frame: np.ndarray, 
                    frame_idx: int = 0) -> List[Detection]:
        """后处理检测结果"""
        detections = []
        
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                # 获取框、置信度、类别
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = boxes.conf[i].cpu().numpy()
                cls_id = int(boxes.cls[i].cpu().numpy())
                label = result.names[cls_id]
                
                # 只保留关注的类别
                if label in self.classes and conf >= self.confidence_threshold:
                    # 调整框到原始图像尺寸
                    h, w = frame.shape[:2]
                    scale_x, scale_y = w / 640, h / 640
                    
                    scaled_bbox = [
                        bbox[0] * scale_x, bbox[1] * scale_y,
                        bbox[2] * scale_x, bbox[3] * scale_y
                    ]
                    
                    detection = Detection(
                        label=label,
                        confidence=float(conf),
                        bbox=scaled_bbox.tolist(),
                        frame_index=frame_idx,
                        timestamp=0.0  # 由外部设置
                    )
                    detections.append(detection)
        
        return detections
    
    def filter_by_class(self, detections: List[Detection], 
                       target_classes: List[str]) -> List[Detection]:
        """按类别过滤检测结果"""
        return [
            det for det in detections 
            if det.label in target_classes
        ]
    
    def visualize_detections(self, frame: np.ndarray, 
                           detections: List[Detection]) -> np.ndarray:
        """可视化检测结果"""
        vis_frame = frame.copy()
        
        for det in detections:
            label = f"{det.label} {det.confidence:.2f}"
            bbox = [int(x) for x in det.bbox]
            
            # 绘制边界框
            color = self._get_color(det.label)
            cv2.rectangle(vis_frame, (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), color, 2)
            
            # 绘制标签
            cv2.putText(vis_frame, label, (bbox[0], bbox[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_frame
    
    def _get_color(self, label: str) -> Tuple[int, int, int]:
        """根据标签获取颜色"""
        color_map = {
            'person': (0, 255, 0),  # 绿色
            'cell phone': (255, 0, 0),  # 蓝色
            'laptop': (0, 0, 255),  # 红色
            'fire': (0, 165, 255),  # 橙色
            'smoke': (128, 128, 128),  # 灰色
        }
        return color_map.get(label, (255, 255, 0))  # 默认黄色