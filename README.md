### 📌 补充说明

1. **截图建议**：在 `outputs/visualizations/` 里放一张分析结果截图，作为 `assets/screenshot.png`，在 README 顶部用 `![screenshot](assets/screenshot.png)` 展示，效果更好。

2. **Badge 徽章**：你可以把项目公开后，用 [shields.io](https://shields.io) 生成更多实时徽章（如 Stars、Last Commit 等）。

3. **演示视频**：如果条件允许，上传一段演示视频到 YouTube/Bilibili，链接放在 README 中，对项目展示非常加分。# huilin-DetectiveEye Inspector

> 🕵️ 零API依赖 · 纯本地运行 · 视频行为智能分析系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GPU](https://img.shields.io/badge/GPU-CUDA%20Optional-orange.svg)](https://developer.nvidia.com/cuda-toolkit)

DetectiveEye Inspector 是一款**完全离线运行**的视频智能分析工具，无需联网、无需API密钥，即可对视频进行目标检测、OCR文字识别和规则匹配，生成带证据的可视化分析报告。

## ✨ 核心特性

- 🔒 **零API依赖**：所有模型本地运行，数据不出本机，保护隐私
- 🧠 **多模态分析**：视觉检测(YOLOv8) + 文字识别(EasyOCR) + 关键词匹配
- ⚡ **智能缓存**：帧级哈希缓存，重复分析同一视频时秒级启动
- 📊 **完整报告**：自动生成 Markdown + JSON 报告，附带证据帧和可视化图表
- 🖥️ **GPU/CPU自适应**：有GPU用GPU，没GPU自动回退CPU
- 🔧 **配置驱动**：YAML配置文件，无需改代码即可调整分析策略

## 🏗️ 架构概览
huilin-DetectiveEye-Inspector/
├── app.py                    # 🎯 总入口，参数解析与流程调度
├── core/
│   ├── detector.py           # 🔍 YOLOv8 目标检测模块
│   ├── ocr_engine.py         # 📝 EasyOCR 文字识别模块
│   ├── rule_engine.py        # 🧠 规则引擎，意图解析与多模态匹配
│   ├── video_processor.py    # 🎬 视频处理与帧提取
│   └── utils.py              # 🔧 工具函数（配置/报告/可视化）
├── config/
│   ├── config.yaml           # ⚙️  主配置文件
│   └── keywords.json         # 📋 关键词/规则模板
├── models/                   # 📦 模型存放目录
├── outputs/                  # 📁 分析结果输出
└── temp/                     # 🗑️ 临时文件（自动清理）

## 🚀 快速开始

### 环境要求

- **Python** >= 3.8
- **FFmpeg**（如未安装会自动下载到 `runtime/ffmpeg/`）
- **内存** >= 4GB（推荐8GB以上）
- **GPU** 可选（有CUDA会自动启用，CPU也能跑）

### 安装

```bash
# 1. 克隆仓库
git clone https://github.com/chene2603-cmd/huilin-DetectiveEye-Inspector.git
cd huilin-DetectiveEye-Inspector

# 2. 安装依赖
pip install -r requirements.txt

# 3. 首次运行（会自动下载模型）
python app.py --video test.mp4 --query "检测玩手机行为"
```

使用示例

```bash
# 基础用法：分析视频中的特定行为
python app.py --video classroom.mp4 --query "有人玩手机或睡觉"

# 指定输出目录
python app.py --video meeting.mp4 --query "检测举手发言" --output outputs/meeting/

# 高性能模式（快速匹配，跳过深度分析）
python app.py --video long_video.mp4 --query "检测异常行为" --fast

# 查看系统信息
python app.py --check-system
```

⚙️ 配置文件说明

编辑 config/config.yaml 即可自定义分析参数：

```yaml
system:
  max_workers: 4          # 并行处理线程数
  temp_cleanup: true      # 分析后自动清理临时文件

models:
  yolo:
    confidence: 0.4       # YOLO检测置信度阈值
    batch_size: 8         # 批处理大小
  ocr:
    languages:            # OCR支持的语言
      - ch_sim            # 简体中文
      - en                # 英文

video:
  preview_fps: 1          # 快速预览采样率
  detail_fps: 6           # 精细分析采样率
  scene_threshold: 0.3    # 场景切换检测阈值
```

📊 输出报告

每次分析会在 outputs/ 下生成一个时间戳目录，包含：

文件/目录 说明
report.md 📝 可读的Markdown分析报告
report.json 📊 结构化的JSON数据
frames/ 🖼️ 所有证据帧截图
visualizations/ 📈 时间线、置信度分布等图表

🗺️ 路线图

· YOLOv8 目标检测
· EasyOCR 文字识别
· 关键词规则匹配
· GPU/CPU 自适应
· 帧级哈希缓存
· 多视频对比分析
· Web可视化界面
· 实时摄像头分析

🤝 贡献

欢迎提交 Issue 和 Pull Request！

📄 许可证

本项目采用 MIT License。

```
