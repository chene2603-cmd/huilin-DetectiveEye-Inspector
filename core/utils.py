"""
工具函数
"""

import os
import json
import yaml
import shutil
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path
import hashlib
import cv2
import numpy as np


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    if not os.path.exists(config_path):
        # 创建默认配置
        default_config = {
            'system': {
                'max_workers': 4,
                'gpu_memory_limit': 0.8,
                'temp_cleanup': True,
                'ffmpeg_path': 'runtime/ffmpeg/ffmpeg'
            },
            'video': {
                'preview_fps': 1,
                'detail_fps': 6,
                'scene_threshold': 0.3
            },
            'models': {
                'yolo': {
                    'model_path': 'models/yolov8n.pt',
                    'confidence': 0.4,
                    'batch_size': 8,
                    'gpu_memory_limit': 0.8
                },
                'ocr': {
                    'languages': ['ch_sim', 'en'],
                    'gpu': True,
                    'batch_size': 16
                }
            },
            'rules': {
                'template_path': 'config/keywords.json'
            },
            'output': {
                'format': 'both',
                'save_frames': True,
                'frame_quality': 0.8
            }
        }
        
        # 确保配置目录存在
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        return default_config
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_config(config: Dict, config_path: str):
    """保存配置文件"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)


def cleanup_temp_files(temp_dir: str = "temp"):
    """清理临时文件"""
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            print(f"🧹 已清理临时目录: {temp_dir}")
        except Exception as e:
            print(f"⚠️  清理临时目录失败: {e}")


def get_file_hash(file_path: str) -> str:
    """计算文件哈希值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def format_time(seconds: float) -> str:
    """格式化时间（秒 -> HH:MM:SS）"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def create_output_directory(video_path: str) -> str:
    """创建输出目录"""
    video_name = Path(video_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/{video_name}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/frames", exist_ok=True)
    os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
    
    return output_dir


def save_evidence_frames(frames: List[np.ndarray], 
                        output_dir: str, 
                        evidence_id: str,
                        quality: int = 85) -> List[str]:
    """保存证据帧"""
    frame_paths = []
    
    for i, frame in enumerate(frames):
        frame_path = f"{output_dir}/frames/{evidence_id}_{i:04d}.jpg"
        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        frame_paths.append(frame_path)
    
    return frame_paths


def generate_report(evidences: List, video_path: str, 
                   query: str, config: Dict) -> Dict:
    """生成分析报告"""
    video_name = Path(video_path).name
    video_hash = get_file_hash(video_path)
    analysis_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 计算统计信息
    total_events = len(evidences)
    
    # 事件类型统计
    event_types = {}
    for evidence in evidences:
        summary = evidence.summary
        event_type = summary.split('|')[0].strip() if '|' in summary else summary
        event_types[event_type] = event_types.get(event_type, 0) + 1
    
    # 时间分布
    timeline = []
    for evidence in evidences:
        timeline.append({
            'start': evidence.time_range[0],
            'end': evidence.time_range[1],
            'summary': evidence.summary,
            'confidence': evidence.confidence
        })
    
    # 创建报告
    report = {
        'metadata': {
            'video_name': video_name,
            'video_hash': video_hash,
            'query': query,
            'analysis_time': analysis_time,
            'analyzer_version': '1.0.0',
            'config': config
        },
        'statistics': {
            'total_events': total_events,
            'event_types': event_types,
            'timeline': timeline
        },
        'evidences': [
            {
                'id': evidence.id,
                'time_range': evidence.time_range,
                'time_range_str': f"{format_time(evidence.time_range[0])} - {format_time(evidence.time_range[1])}",
                'summary': evidence.summary,
                'description': evidence.description,
                'confidence': evidence.confidence,
                'detection_count': len(evidence.detections),
                'ocr_count': len(evidence.ocr_results),
                'frame_count': len(evidence.frame_paths),
                'frame_paths': evidence.frame_paths
            }
            for evidence in evidences
        ]
    }
    
    return report


def save_report_markdown(report: Dict, output_dir: str):
    """保存Markdown报告"""
    md_path = f"{output_dir}/report.md"
    
    with open(md_path, 'w', encoding='utf-8') as f:
        # 标题
        f.write("# 视频分析报告\n\n")
        
        # 基本信息
        f.write("## 📋 基本信息\n\n")
        f.write(f"- **视频文件**: {report['metadata']['video_name']}\n")
        f.write(f"- **分析时间**: {report['metadata']['analysis_time']}\n")
        f.write(f"- **查询问题**: {report['metadata']['query']}\n")
        f.write(f"- **文件哈希**: {report['metadata']['video_hash'][:8]}...\n")
        f.write(f"- **总事件数**: {report['statistics']['total_events']} 个\n")
        
        # 事件类型统计
        f.write("\n## 📊 事件类型统计\n\n")
        for event_type, count in report['statistics']['event_types'].items():
            f.write(f"- {event_type}: {count} 次\n")
        
        # 时间线分布
        f.write("\n## ⏰ 时间线分布\n\n")
        f.write("| 开始时间 | 结束时间 | 事件摘要 | 置信度 |\n")
        f.write("|----------|----------|----------|--------|\n")
        
        for item in report['statistics']['timeline']:
            start_str = format_time(item['start'])
            end_str = format_time(item['end'])
            summary = item['summary'][:20] + "..." if len(item['summary']) > 20 else item['summary']
            confidence = item['confidence']
            
            f.write(f"| {start_str} | {end_str} | {summary} | {confidence:.2f} |\n")
        
        # 详细证据
        f.write("\n## 🔍 详细证据\n\n")
        for i, evidence in enumerate(report['evidences']):
            f.write(f"### 事件 {i+1}: {evidence['summary']}\n\n")
            f.write(f"- **ID**: {evidence['id']}\n")
            f.write(f"- **时间**: {evidence['time_range_str']} (持续 {evidence['time_range'][1] - evidence['time_range'][0]:.1f} 秒)\n")
            f.write(f"- **置信度**: {evidence['confidence']:.2f}\n")
            f.write(f"- **视觉检测**: {evidence['detection_count']} 个目标\n")
            f.write(f"- **文字识别**: {evidence['ocr_count']} 条文字\n")
            f.write(f"- **关键帧**: {evidence['frame_count']} 张\n")
            f.write(f"- **描述**: {evidence['description']}\n\n")
            
            # 显示关键帧
            if evidence['frame_paths']:
                f.write("**关键帧**:\n\n")
                for frame_path in evidence['frame_paths'][:3]:  # 最多显示3张
                    frame_name = Path(frame_path).name
                    f.write(f"![关键帧]({frame_name})\n\n")
            
            f.write("---\n\n")
        
        # 系统信息
        f.write("## ⚙️ 系统信息\n\n")
        f.write(f"- **分析器版本**: {report['metadata']['analyzer_version']}\n")
        f.write(f"- **YOLO置信度**: {report['metadata']['config']['models']['yolo']['confidence']}\n")
        f.write(f"- **OCR语言**: {', '.join(report['metadata']['config']['models']['ocr']['languages'])}\n")
        
        # 附件列表
        f.write("\n## 📎 附件列表\n\n")
        f.write("- `report.json` - 完整JSON数据\n")
        f.write("- `frames/` - 所有证据帧\n")
        f.write("- `visualizations/` - 可视化结果\n")
    
    print(f"✅ 已保存Markdown报告: {md_path}")
    return md_path


def save_report_json(report: Dict, output_dir: str):
    """保存JSON报告"""
    json_path = f"{output_dir}/report.json"
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 已保存JSON报告: {json_path}")
    return json_path


def create_visualization(report: Dict, output_dir: str):
    """创建可视化图表"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # 非交互模式
        
        evidences = report['evidences']
        if not evidences:
            return None
        
        # 1. 时间线图
        plt.figure(figsize=(12, 6))
        
        for i, evidence in enumerate(evidences):
            start = evidence['time_range'][0]
            end = evidence['time_range'][1]
            confidence = evidence['confidence']
            
            # 使用颜色表示置信度
            color = plt.cm.RdYlGn(confidence)  # 红-黄-绿色系
            plt.hlines(y=i, xmin=start, xmax=end, 
                      colors=[color], linewidth=10, alpha=0.7)
            
            # 添加标签
            plt.text(start, i, evidence['summary'][:15] + "...", 
                    va='center', ha='left', fontsize=8)
        
        plt.xlabel('时间 (秒)')
        plt.ylabel('事件')
        plt.title('事件时间线分布')
        plt.tight_layout()
        timeline_path = f"{output_dir}/visualizations/timeline.png"
        plt.savefig(timeline_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. 置信度分布图
        plt.figure(figsize=(8, 5))
        confidences = [e['confidence'] for e in evidences]
        plt.hist(confidences, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('置信度')
        plt.ylabel('事件数量')
        plt.title('置信度分布')
        plt.grid(True, alpha=0.3)
        conf_path = f"{output_dir}/visualizations/confidence_dist.png"
        plt.savefig(conf_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. 事件类型饼图
        plt.figure(figsize=(8, 8))
        event_types = report['statistics']['event_types']
        
        if event_types:
            labels = list(event_types.keys())
            sizes = list(event_types.values())
            
            # 自动调整标签位置避免重叠
            def autopct_format(pct):
                return f'{pct:.1f}%' if pct >= 3 else ''
            
            plt.pie(sizes, labels=labels, autopct=autopct_format,
                   startangle=90, textprops={'fontsize': 9})
            plt.axis('equal')
            plt.title('事件类型分布')
            pie_path = f"{output_dir}/visualizations/event_types.png"
            plt.savefig(pie_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        return {
            'timeline': timeline_path,
            'confidence': conf_path,
            'event_types': pie_path if event_types else None
        }
        
    except ImportError:
        print("⚠️  Matplotlib未安装，跳过可视化生成")
        return None
    except Exception as e:
        print(f"⚠️  可视化生成失败: {e}")
        return None


def create_evidence_collage(frame_paths: List[str], output_path: str, 
                          cols: int = 4, max_frames: int = 12):
    """创建证据帧拼图"""
    if not frame_paths:
        return None
    
    # 限制帧数
    frame_paths = frame_paths[:min(len(frame_paths), max_frames)]
    
    try:
        # 读取所有帧
        frames = []
        for path in frame_paths:
            if os.path.exists(path):
                frame = cv2.imread(path)
                if frame is not None:
                    # 统一大小
                    frame_resized = cv2.resize(frame, (320, 240))
                    frames.append(frame_resized)
        
        if not frames:
            return None
        
        # 计算拼图布局
        rows = (len(frames) + cols - 1) // cols
        
        # 创建画布
        collage = np.zeros((rows * 240, cols * 320, 3), dtype=np.uint8)
        
        # 填充拼图
        for i, frame in enumerate(frames):
            row = i // cols
            col = i % cols
            collage[row*240:(row+1)*240, col*320:(col+1)*320] = frame
            
            # 添加序号
            cv2.putText(collage, f"{i+1}", 
                       (col*320 + 10, row*240 + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 保存拼图
        cv2.imwrite(output_path, collage, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return output_path
        
    except Exception as e:
        print(f"⚠️  创建证据拼图失败: {e}")
        return None


def compress_output(output_dir: str, format: str = 'zip'):
    """压缩输出文件（目前仅支持zip格式）"""
    if format != 'zip':
        print(f"⚠️  暂不支持 {format} 格式，将使用 zip 压缩")
    
    try:
        import zipfile
        
        output_name = Path(output_dir).name
        zip_path = f"{output_dir}.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname=arcname)
        
        print(f"✅ 已压缩输出文件: {zip_path}")
        return zip_path
        
    except Exception as e:
        print(f"❌ 压缩失败: {e}")
        return None


def setup_logging(log_dir: str = "logs"):
    """设置日志"""
    import logging
    
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = f"{log_dir}/detective_eye_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('DetectiveEye')


def check_system_requirements():
    """检查系统要求"""
    requirements = {
        'python_version': '>=3.8',
        'ffmpeg': True,
        'cuda': False,
        'memory': '>=4GB'
    }
    
    issues = []
    
    # 检查Python版本
    import sys
    if sys.version_info < (3, 8):
        issues.append(f"Python版本需要>=3.8，当前版本: {sys.version}")
    
    # 检查FFmpeg
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            issues.append("FFmpeg未找到或不可用")
    except:
        issues.append("FFmpeg未找到")
    
    # 检查CUDA
    try:
        import torch
        if torch.cuda.is_available():
            requirements['cuda'] = True
    except:
        pass
    
    # 检查内存
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 4:
            issues.append(f"系统内存不足: {memory_gb:.1f}GB，推荐>=4GB")
    except:
        pass
    
    return requirements, issues


def get_system_info() -> Dict:
    """获取系统信息"""
    import platform
    import psutil
    
    info = {
        'system': {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
        },
        'memory': {
            'total_gb': psutil.virtual_memory().total / (1024**3),
            'available_gb': psutil.virtual_memory().available / (1024**3),
        },
        'disk': {
            'free_gb': psutil.disk_usage('.').free / (1024**3),
        }
    }
    
    # GPU信息
    try:
        import torch
        if torch.cuda.is_available():
            info['gpu'] = {
                'available': True,
                'device_name': torch.cuda.get_device_name(0),
                'memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3)
            }
        else:
            info['gpu'] = {'available': False}
    except:
        info['gpu'] = {'available': False, 'error': 'torch not available'}
    
    return info


def print_system_info():
    """打印系统信息"""
    info = get_system_info()
    
    print("=" * 50)
    print("🔧 系统信息")
    print("=" * 50)
    
    print(f"操作系统: {info['system']['platform']}")
    print(f"处理器: {info['system']['processor']}")
    print(f"Python版本: {info['system']['python_version']}")
    print(f"内存: {info['memory']['total_gb']:.1f}GB (可用: {info['memory']['available_gb']:.1f}GB)")
    print(f"磁盘空间: {info['disk']['free_gb']:.1f}GB 可用")
    
    if info['gpu'].get('available'):
        print(f"GPU: {info['gpu']['device_name']}")
        print(f"GPU显存: {info['gpu']['memory_gb']:.1f}GB")
    else:
        print("GPU: 未找到或不可用")
    
    print("=" * 50)