"""
文字识别层 - EasyOCR离线OCR
"""

import easyocr
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import threading
from pathlib import Path
import time

@dataclass
class OCRResult:
    """OCR识别结果"""
    text: str
    confidence: float
    bbox: List[Tuple[float, float]]  # 四个顶点
    timestamp: float


class OCREngine:
    """OCR引擎"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.ocr_config = config.get('models', {}).get('ocr', {})
        
        # 语言配置
        self.languages = self.ocr_config.get('languages', ['ch_sim', 'en'])
        
        # 设备配置
        self.gpu = self.ocr_config.get('gpu', True) and self._check_gpu()
        
        # 初始化EasyOCR
        print("🔤 初始化EasyOCR...")
        self.reader = self._init_reader()
        
        # 识别区域策略
        self.region_strategies = [
            self._recognize_screen_regions,  # 屏幕区域优先
            self._recognize_bottom_third,    # 底部字幕区
            self._recognize_whole_frame       # 全帧识别
        ]
        
        # 缓存
        self.cache = {}
        self.cache_lock = threading.Lock()
    
    def _check_gpu(self) -> bool:
        """检查GPU可用性"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def _init_reader(self):
        """初始化EasyOCR阅读器"""
        # 检查模型是否存在
        model_dir = Path.home() / '.EasyOCR' / 'model'
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置EasyOCR参数
        reader_params = {
            'lang_list': self.languages,
            'gpu': self.gpu,
            'model_storage_directory': str(model_dir),
            'download_enabled': True,  # 自动下载模型
            'verbose': False
        }
        
        # 添加识别器参数
        recognizer_params = {
            'batch_size': self.ocr_config.get('batch_size', 16),
            'height': 64,
            'width': 256
        }
        
        reader_params['recognizer'] = recognizer_params
        
        return easyocr.Reader(**reader_params)
    
    def recognize(self, frame: np.ndarray) -> List[OCRResult]:
        """识别单帧文字"""
        # 检查缓存
        cache_key = hash(frame.data.tobytes())
        with self.cache_lock:
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        results = []
        
        # 按策略顺序识别
        for strategy in self.region_strategies:
            strategy_results = strategy(frame)
            if strategy_results:
                results.extend(strategy_results)
                break  # 如果某个策略成功，跳过其他策略
        
        # 后处理
        processed_results = self._postprocess(results)
        
        # 缓存结果
        with self.cache_lock:
            self.cache[cache_key] = processed_results
        
        return processed_results
    
    def recognize_batch(self, frames: List[np.ndarray]) -> List[List[OCRResult]]:
        """批量识别文字"""
        all_results = []
        
        batch_size = self.ocr_config.get('batch_size', 16)
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            batch_results = []
            
            for frame in batch_frames:
                results = self.recognize(frame)
                batch_results.append(results)
            
            all_results.extend(batch_results)
        
        return all_results
    
    def _recognize_screen_regions(self, frame: np.ndarray) -> List[OCRResult]:
        """识别屏幕/标牌区域"""
        # 这里可以集成YOLO检测到的屏幕区域
        # 暂时返回空列表，由外部提供检测结果
        return []
    
    def _recognize_bottom_third(self, frame: np.ndarray) -> List[OCRResult]:
        """识别底部1/3区域（字幕区）"""
        h, w = frame.shape[:2]
        bottom_third = frame[int(2*h/3):, :]
        
        if bottom_third.size == 0:
            return []
        
        # 增强对比度
        gray = cv2.cvtColor(bottom_third, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.equalizeHist(gray)
        
        # 识别文字
        results = self.reader.readtext(
            enhanced,
            paragraph=False,
            width_ths=0.7,
            height_ths=0.7
        )
        
        ocr_results = []
        for (bbox, text, confidence) in results:
            if confidence > 0.3:  # 置信度阈值
                # 调整bbox坐标到原始图像
                adjusted_bbox = []
                for (x, y) in bbox:
                    adjusted_bbox.append((x, y + 2*h/3))
                
                ocr_result = OCRResult(
                    text=text,
                    confidence=confidence,
                    bbox=adjusted_bbox,
                    timestamp=0.0
                )
                ocr_results.append(ocr_result)
        
        return ocr_results
    
    def _recognize_whole_frame(self, frame: np.ndarray) -> List[OCRResult]:
        """全帧识别"""
        # 缩小图像以加快识别速度
        h, w = frame.shape[:2]
        scale = 0.5
        small_frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        
        # 识别文字
        results = self.reader.readtext(
            small_frame,
            paragraph=True,
            width_ths=0.5,
            height_ths=0.5
        )
        
        ocr_results = []
        for (bbox, text, confidence) in results:
            if confidence > 0.2:  # 全帧识别使用更低的阈值
                # 调整bbox坐标到原始图像
                adjusted_bbox = []
                for (x, y) in bbox:
                    adjusted_bbox.append((x/scale, y/scale))
                
                ocr_result = OCRResult(
                    text=text,
                    confidence=confidence,
                    bbox=adjusted_bbox,
                    timestamp=0.0
                )
                ocr_results.append(ocr_result)
        
        return ocr_results
    
    def _postprocess(self, results: List[OCRResult]) -> List[OCRResult]:
        """OCR结果后处理"""
        if not results:
            return []
        
        # 1. 去重：合并相似的文本
        unique_results = []
        seen_texts = set()
        
        for result in results:
            # 简单去重：完全相同的文本
            if result.text in seen_texts:
                continue
            
            # 相似文本检测（编辑距离）
            is_duplicate = False
            for seen_text in seen_texts:
                if self._text_similarity(result.text, seen_text) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_texts.add(result.text)
                unique_results.append(result)
        
        # 2. 过滤无效文本
        filtered_results = []
        for result in unique_results:
            text = result.text.strip()
            
            # 过滤太短的文本
            if len(text) < 2:
                continue
            
            # 过滤纯符号
            if not any(c.isalnum() for c in text):
                continue
            
            # 过滤常见噪声
            noise_patterns = ['。', '.', ',', '，', '!', '！', '?', '？']
            if all(c in noise_patterns for c in text):
                continue
            
            filtered_results.append(result)
        
        return filtered_results
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（基于编辑距离）"""
        # 简单的相似度计算
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1, text2).ratio()
    
    def extract_keywords(self, ocr_results: List[OCRResult], 
                        keywords: List[str]) -> List[OCRResult]:
        """提取包含关键词的OCR结果"""
        keyword_results = []
        
        for result in ocr_results:
            text_lower = result.text.lower()
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    keyword_results.append(result)
                    break
        
        return keyword_results
    
    def visualize_ocr(self, frame: np.ndarray, 
                    ocr_results: List[OCRResult]) -> np.ndarray:
        """可视化OCR结果"""
        vis_frame = frame.copy()
        
        for result in ocr_results:
            bbox = result.bbox
            text = result.text
            confidence = result.confidence
            
            # 将bbox转换为整数坐标
            points = np.array(bbox, dtype=np.int32)
            
            # 绘制文本框
            cv2.polylines(vis_frame, [points], True, (0, 255, 0), 2)
            
            # 绘制文本背景
            text_position = (int(points[0][0]), int(points[0][1] - 5))
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            cv2.rectangle(vis_frame,
                         (text_position[0], text_position[1] - text_height - 5),
                         (text_position[0] + text_width, text_position[1] + 5),
                         (0, 255, 0), -1)
            
            # 绘制文本
            cv2.putText(vis_frame, f"{text} ({confidence:.2f})",
                       text_position,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return vis_frame