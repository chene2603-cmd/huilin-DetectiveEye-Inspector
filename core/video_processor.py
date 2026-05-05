"""
视频底层层 - 处理视频输入和分段
"""

import subprocess
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import tempfile
import os

@dataclass
class VideoSegment:
    """视频段落"""
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    scene_type: str = "normal"  # normal, scene_change, silence
    keyframes: List[str] = None
    
    def __post_init__(self):
        if self.keyframes is None:
            self.keyframes = []


class VideoProcessor:
    """视频处理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.video_config = config.get('video', {})
        self.system_config = config.get('system', {})
        
        # FFmpeg路径
        self.ffmpeg_path = self.system_config.get(
            'ffmpeg_path', 
            'runtime/ffmpeg/ffmpeg'
        )
        
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp(prefix="detective_eye_")
        print(f"📁 临时目录: {self.temp_dir}")
    
    def get_video_info(self, video_path: str) -> Dict:
        """获取视频信息"""
        cmd = [
            self.ffmpeg_path, '-i', video_path
        ]
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=False
            )
            
            # 解析视频信息
            info = {
                'path': video_path,
                'name': Path(video_path).name,
                'size': os.path.getsize(video_path)
            }
            
            # 解析时长
            for line in result.stderr.split('\n'):
                if 'Duration:' in line:
                    duration_str = line.split('Duration: ')[1].split(',')[0]
                    h, m, s = duration_str.split(':')
                    info['duration'] = int(h)*3600 + int(m)*60 + float(s)
                elif 'Stream #' in line and 'Video:' in line:
                    if 'fps' in line:
                        fps_str = line.split('fps')[0].split(',')[-1].strip()
                        info['fps'] = float(fps_str)
                    if 'x' in line:
                        res_str = line.split('Video:')[1].split()[2]
                        w, h = res_str.split('x')
                        info['width'] = int(w)
                        info['height'] = int(h)
            
            return info
            
        except Exception as e:
            print(f"❌ 获取视频信息失败: {e}")
            return {}
    
    def segment(self, video_path: str) -> List[VideoSegment]:
        """
        视频分段处理
        1. 全局概览（1 FPS）
        2. 场景边界检测
        3. 音频静默检测
        """
        print("🎬 开始视频分段...")
        
        # 1. 获取视频信息
        video_info = self.get_video_info(video_path)
        duration = video_info.get('duration', 0)
        fps = video_info.get('fps', 30)
        
        if duration == 0:
            raise ValueError("无法获取视频时长")
        
        # 2. 提取关键帧（1 FPS）
        preview_fps = self.video_config.get('preview_fps', 1)
        frame_count = int(duration * preview_fps)
        
        print(f"📊 提取 {frame_count} 个预览帧...")
        frames = self._extract_preview_frames(video_path, preview_fps, frame_count)
        
        # 3. 场景边界检测
        print("🔍 检测场景边界...")
        scene_changes = self._detect_scene_changes(frames)
        
        # 4. 音频静默检测
        print("🔇 检测音频静默...")
        silence_segments = self._detect_silence(video_path)
        
        # 5. 合并分段
        segments = self._merge_segments(
            duration, fps, scene_changes, silence_segments
        )
        
        # 6. 为每个分段提取关键帧
        print("🖼️ 提取分段关键帧...")
        for seg in segments:
            seg.keyframes = self._extract_segment_keyframes(video_path, seg)
        
        print(f"✅ 视频分段完成: {len(segments)} 个段落")
        return segments
    
    def _extract_preview_frames(self, video_path: str, 
                              target_fps: int, 
                              frame_count: int) -> List[np.ndarray]:
        """提取预览帧"""
        frames = []
        
        # 使用OpenCV提取
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        
        if original_fps <= 0:
            original_fps = 30
        
        # 计算跳帧间隔
        frame_interval = int(original_fps / target_fps)
        
        frame_idx = 0
        while len(frames) < frame_count:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                # 缩放到统一大小以节省内存
                frame_small = cv2.resize(frame, (320, 240))
                frames.append(frame_small)
            
            frame_idx += 1
        
        cap.release()
        return frames
    
    def _detect_scene_changes(self, frames: List[np.ndarray]) -> List[int]:
        """检测场景变化"""
        scene_changes = [0]  # 从第0帧开始
        
        threshold = self.video_config.get('scene_threshold', 0.3)
        
        for i in range(1, len(frames)):
            if i >= len(frames):
                break
                
            # 计算帧间差异
            prev_frame = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
            curr_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            # 使用结构相似性指数
            diff = self._frame_difference(prev_frame, curr_frame)
            
            if diff > threshold:
                scene_changes.append(i)
        
        return scene_changes
    
    def _frame_difference(self, frame1: np.ndarray, 
                         frame2: np.ndarray) -> float:
        """计算帧间差异度"""
        # 直方图比较
        hist1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])
        
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        
        # 相关性
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # 转换为差异度 (0-1, 1表示完全不同)
        diff = 1.0 - (correlation + 1) / 2
        
        return diff
    
    def _detect_silence(self, video_path: str) -> List[Tuple[float, float]]:
        """检测音频静默段"""
        silence_segments = []
        
        # 提取音频
        audio_path = os.path.join(self.temp_dir, "audio.wav")
        cmd = [
            self.ffmpeg_path, '-i', video_path,
            '-vn', '-ac', '1', '-ar', '16000',
            '-acodec', 'pcm_s16le',
            audio_path, '-y'
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            
            # 读取音频并检测静默
            import wave
            import numpy as np
            
            with wave.open(audio_path, 'rb') as wav:
                sample_rate = wav.getframerate()
                n_frames = wav.getnframes()
                audio_data = wav.readframes(n_frames)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # 简单的静默检测
                threshold = 500  # 静默阈值
                window_size = sample_rate // 10  # 100ms窗口
                
                for i in range(0, len(audio_array), window_size):
                    window = audio_array[i:i+window_size]
                    if np.max(np.abs(window)) < threshold:
                        # 静默段
                        start_time = i / sample_rate
                        end_time = (i + window_size) / sample_rate
                        silence_segments.append((start_time, end_time))
                        
        except Exception as e:
            print(f"⚠️  音频静默检测失败: {e}")
        
        return silence_segments
    
    def _merge_segments(self, duration: float, fps: float,
                       scene_changes: List[int], 
                       silence_segments: List[Tuple[float, float]]) -> List[VideoSegment]:
        """合并生成最终分段"""
        segments = []
        
        # 将场景变化点转换为时间
        scene_times = [idx / 1.0 for idx in scene_changes]  # 预览帧率是1fps
        
        # 添加静默段边界
        all_boundaries = sorted(scene_times + 
                               [t for seg in silence_segments for t in seg])
        
        # 去重并排序
        all_boundaries = sorted(set(all_boundaries))
        
        # 确保以0开始，以duration结束
        if 0 not in all_boundaries:
            all_boundaries.insert(0, 0)
        if duration not in all_boundaries:
            all_boundaries.append(duration)
        
        # 创建分段
        for i in range(len(all_boundaries) - 1):
            start = all_boundaries[i]
            end = all_boundaries[i + 1]
            
            # 跳过太短的分段（小于2秒）
            if end - start < 2.0:
                continue
            
            segment = VideoSegment(
                start_time=start,
                end_time=end,
                start_frame=int(start * fps),
                end_frame=int(end * fps)
            )
            
            # 判断分段类型
            for silence_start, silence_end in silence_segments:
                if start >= silence_start and end <= silence_end:
                    segment.scene_type = "silence"
                    break
            
            segments.append(segment)
        
        return segments
    
    def _extract_segment_keyframes(self, video_path: str, 
                                 segment: VideoSegment) -> List[str]:
        """为分段提取关键帧"""
        keyframes = []
        
        # 提取分段中间的关键帧
        mid_time = (segment.start_time + segment.end_time) / 2
        
        output_path = os.path.join(
            self.temp_dir,
            f"segment_{segment.start_time:.1f}_{segment.end_time:.1f}.jpg"
        )
        
        cmd = [
            self.ffmpeg_path, '-i', video_path,
            '-ss', str(mid_time),  # 跳转到指定时间
            '-vframes', '1',  # 提取1帧
            '-q:v', '2',  # 高质量
            output_path, '-y'
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            keyframes.append(output_path)
        except Exception as e:
            print(f"⚠️  提取关键帧失败: {e}")
        
        return keyframes
    
    def extract_frames(self, video_path: str, segment: VideoSegment,
                      high_fps: bool = False) -> List[np.ndarray]:
        """
        提取分段内的视频帧
        
        Args:
            video_path: 视频文件路径
            segment: 视频段落
            high_fps: 是否使用高帧率模式
        Returns:
            视频帧列表
        """
        frames = []
        
        # 确定帧率
        target_fps = self.video_config.get(
            'detail_fps' if high_fps else 'preview_fps', 
            6 if high_fps else 1
        )
        
        # 计算需要提取的帧数
        segment_duration = segment.end_time - segment.start_time
        frame_count = int(segment_duration * target_fps)
        
        if frame_count == 0:
            frame_count = 1
        
        # 使用FFmpeg提取指定时间段的帧
        output_pattern = os.path.join(
            self.temp_dir,
            f"frames_{segment.start_time:.1f}_%04d.jpg"
        )
        
        cmd = [
            self.ffmpeg_path, 
            '-i', video_path,
            '-ss', str(segment.start_time),  # 开始时间
            '-t', str(segment_duration),  # 持续时间
            '-vf', f'fps={target_fps}',  # 帧率
            '-q:v', '2',  # 质量
            output_pattern, '-y'
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            
            # 读取提取的帧
            for i in range(1, frame_count + 1):
                frame_path = output_pattern.replace('%04d', f'{i:04d}')
                if os.path.exists(frame_path):
                    frame = cv2.imread(frame_path)
                    if frame is not None:
                        frames.append(frame)
                    # 清理临时文件
                    os.remove(frame_path)
                    
        except Exception as e:
            print(f"❌ 提取帧失败: {e}")
        
        return frames
    
    def cleanup(self):
        """清理临时文件"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"🧹 清理临时目录: {self.temp_dir}")