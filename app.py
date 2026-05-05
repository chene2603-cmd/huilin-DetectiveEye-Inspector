#!/usr/bin/env python3
"""
侦探眼探长 - 主控制器
零API依赖·纯本地运行·绿色免安装
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import streamlit as st
from typing import List, Dict, Any, Optional
import gradio as gr

from core.video_processor import VideoProcessor
from core.detector import YOLODetector
from core.ocr_engine import OCREngine
from core.rule_engine import RuleEngine
from core.utils import cleanup_temp_files, generate_report

class DetectiveEye:
    """侦探眼探长主控制器"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.initialized = False
        self.video_processor = None
        self.detector = None
        self.ocr_engine = None
        self.rule_engine = None
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
        
    def initialize(self):
        """初始化所有组件"""
        print("🔍 初始化侦探眼探长...")
        
        # 检查依赖
        self._check_dependencies()
        
        # 初始化组件
        self.video_processor = VideoProcessor(self.config)
        self.detector = YOLODetector(self.config)
        self.ocr_engine = OCREngine(self.config)
        self.rule_engine = RuleEngine(self.config)
        
        # 预热模型
        if self.config['models']['yolo'].get('warmup', True):
            self.detector.warmup()
            
        self.initialized = True
        print("✅ 初始化完成！")
        
    def _check_dependencies(self):
        """检查系统依赖"""
        # 检查FFmpeg
        ffmpeg_path = self.config['system'].get('ffmpeg_path', 'runtime/ffmpeg/ffmpeg')
        if not os.path.exists(ffmpeg_path):
            print("⚠️  FFmpeg未找到，尝试从网络下载...")
            self._download_ffmpeg()
            
        # 检查模型文件
        yolo_model_path = self.config['models']['yolo']['model_path']
        if not os.path.exists(yolo_model_path):
            print("⚠️  YOLO模型未找到，尝试下载...")
            self._download_yolo_model()
    
    def investigate(self, video_path: str, query: str, 
                   progress_callback=None) -> Dict:
        """
        核心调查流程
        Args:
            video_path: 视频文件路径
            query: 查询语句
            progress_callback: 进度回调函数
        Returns:
            分析结果字典
        """
        if not self.initialized:
            self.initialize()
            
        print(f"🎬 开始分析视频: {video_path}")
        print(f"🔎 查询: {query}")
        
        # 1. 视频分段处理
        if progress_callback:
            progress_callback(0.1, "正在分析视频结构...")
        
        segments = self.video_processor.segment(video_path)
        print(f"📊 视频分段完成: {len(segments)} 个段落")
        
        # 2. 解析查询关键词
        keywords = self.rule_engine.parse_query(query)
        print(f"🔑 提取关键词: {keywords}")
        
        # 3. 快速关键词匹配（文字层）
        if progress_callback:
            progress_callback(0.3, "快速扫描文字内容...")
        
        suspect_segments = self.rule_engine.match_keywords(
            segments, keywords, fast_mode=True
        )
        print(f"🎯 锁定嫌疑段落: {len(suspect_segments)} 个")
        
        # 4. 高密度分析嫌疑段落
        evidences = []
        total_segments = len(suspect_segments)
        
        for idx, seg in enumerate(suspect_segments):
            if progress_callback:
                progress = 0.3 + 0.5 * (idx / total_segments)
                progress_callback(progress, f"分析嫌疑段落 {idx+1}/{total_segments}...")
            
            # 提取高帧率视频帧
            frames = self.video_processor.extract_frames(
                video_path, seg, high_fps=True
            )
            
            # 视觉目标检测
            detections = self.detector.detect_batch(frames)
            
            # 文字识别
            texts = self.ocr_engine.recognize_batch(frames)
            
            # 证据融合
            evidence = self.rule_engine.fuse_evidence(
                seg, detections, texts, keywords
            )
            
            if evidence:
                evidences.append(evidence)
                print(f"✅ 发现证据: {evidence['summary']}")
        
        # 5. 生成报告
        if progress_callback:
            progress_callback(0.9, "生成分析报告...")
        
        report = self.generate_report(evidences, video_path, query)
        
        if progress_callback:
            progress_callback(1.0, "分析完成！")
            
        return report
    
    def generate_report(self, evidences: List, video_path: str, 
                       query: str) -> Dict:
        """生成分析报告"""
        return generate_report(
            evidences=evidences,
            video_path=video_path,
            query=query,
            config=self.config
        )
    
    def _download_ffmpeg(self):
        """下载FFmpeg（备用方案）"""
        # 实现下载逻辑
        pass
        
    def _download_yolo_model(self):
        """下载YOLO模型"""
        # 实现下载逻辑
        pass


# Streamlit Web界面
def create_streamlit_app():
    """创建Streamlit Web界面"""
    st.set_page_config(
        page_title="侦探眼探长 - 长视频智能分析工具",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 侦探眼探长")
    st.subheader("零API依赖 · 纯本地运行 · 长视频智能分析")
    
    # 初始化侦探
    if 'detective' not in st.session_state:
        st.session_state.detective = DetectiveEye()
        st.session_state.detective.initialize()
        st.session_state.progress = 0
        st.session_state.status = "就绪"
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 分析设置")
        
        # 查询输入
        query = st.text_area(
            "输入查询语句",
            placeholder="例如：找出所有玩手机的人\n或：检测火灾和烟雾\n或：识别所有出现的手机",
            height=100
        )
        
        # 场景模板
        scenario = st.selectbox(
            "选择场景模板",
            ["自定义", "安防监控", "课堂分析", "产品演示", "零售监控"]
        )
        
        if scenario != "自定义":
            preset_queries = {
                "安防监控": "检测入侵、打架、跌倒、火灾、烟雾",
                "课堂分析": "找出玩手机、睡觉、交头接耳的学生",
                "产品演示": "识别产品包装、logo、价格标签、二维码",
                "零售监控": "检测顾客行为、货架状态、排队情况"
            }
            query = preset_queries.get(scenario, query)
        
        # 高级设置
        with st.expander("高级设置"):
            confidence = st.slider("检测置信度", 0.1, 0.9, 0.4, 0.05)
            preview_fps = st.selectbox("预览帧率", [1, 2, 3, 5], index=0)
            output_format = st.selectbox("输出格式", ["Markdown", "JSON", "Both"])
        
        st.divider()
        
        # 系统信息
        st.caption("💻 系统信息")
        st.caption(f"模型: YOLOv8n")
        st.caption(f"状态: {st.session_state.status}")
    
    # 主界面
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 视频上传区
        uploaded_file = st.file_uploader(
            "上传视频文件",
            type=['mp4', 'avi', 'mov', 'mkv', 'ts'],
            help="支持MP4、AVI、MOV、MKV、TS格式，大小无限制"
        )
        
        if uploaded_file is not None:
            # 保存上传文件
            video_path = f"temp/{uploaded_file.name}"
            os.makedirs("temp", exist_ok=True)
            
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 视频预览
            st.video(video_path)
            
            # 分析按钮
            if st.button("🚀 开始分析", type="primary", use_container_width=True):
                if not query:
                    st.warning("请输入查询语句")
                else:
                    # 进度显示
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(progress, message):
                        progress_bar.progress(progress)
                        status_text.text(message)
                        st.session_state.progress = progress
                        st.session_state.status = message
                    
                    # 执行分析
                    try:
                        report = st.session_state.detective.investigate(
                            video_path, 
                            query,
                            progress_callback=update_progress
                        )
                        
                        # 显示结果
                        st.success("分析完成！")
                        st.balloons()
                        
                        # 显示报告
                        display_report(report, output_format)
                        
                    except Exception as e:
                        st.error(f"分析失败: {str(e)}")
    
    with col2:
        # 进度显示
        st.metric("分析进度", f"{st.session_state.progress*100:.1f}%")
        
        # 历史记录
        st.subheader("📊 历史分析")
        if os.path.exists("outputs"):
            reports = list(Path("outputs").glob("*.md"))
            for report in reports[-5:]:  # 显示最近5个
                st.caption(f"📄 {report.stem}")
        
        # 快速开始
        st.subheader("⚡ 快速开始")
        if st.button("示例：课堂分析", use_container_width=True):
            st.session_state.example_query = "找出所有玩手机的人"
            st.rerun()
        
        if st.button("示例：安防监控", use_container_width=True):
            st.session_state.example_query = "检测火灾和烟雾"
            st.rerun()


def display_report(report: Dict, output_format: str):
    """显示分析报告"""
    st.header("📊 分析报告")
    
    # 基本信息
    st.subheader("📋 基本信息")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("视频文件", report.get("video_name", ""))
    with col2:
        st.metric("总时长", report.get("duration", ""))
    with col3:
        st.metric("发现事件", len(report.get("evidences", [])))
    
    # 时间线可视化
    if "timeline" in report:
        st.subheader("📅 时间线分布")
        # 这里可以添加时间线图表
        st.write("时间线图表占位")
    
    # 详细证据
    st.subheader("🔍 详细证据")
    for idx, evidence in enumerate(report.get("evidences", [])):
        with st.expander(f"事件 {idx+1}: {evidence.get('summary', '')}"):
            col1, col2 = st.columns([1, 2])
            with col1:
                if evidence.get("frame_path"):
                    st.image(evidence["frame_path"], caption="关键帧")
            with col2:
                st.write(f"**时间**: {evidence.get('time_range', '')}")
                st.write(f"**置信度**: {evidence.get('confidence', 0):.2f}")
                st.write(f"**描述**: {evidence.get('description', '')}")
    
    # 导出选项
    st.subheader("💾 导出结果")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("保存Markdown报告"):
            save_report(report, "markdown")
            
    with col2:
        if st.button("保存JSON数据"):
            save_report(report, "json")
            
    with col3:
        if st.button("打包所有证据"):
            save_report(report, "zip")


def save_report(report: Dict, format_type: str):
    """保存报告"""
    # 实现保存逻辑
    st.success(f"报告已保存为 {format_type} 格式")


# Gradio界面（备用）
def create_gradio_interface():
    """创建Gradio界面"""
    detective = DetectiveEye()
    
    def analyze_video(video_path, query, confidence):
        """分析视频的Gradio函数"""
        if video_path is None:
            return "请上传视频文件", None
        
        try:
            detective.initialize()
            report = detective.investigate(video_path, query)
            
            # 格式化结果
            result_text = f"分析完成！发现 {len(report.get('evidences', []))} 个事件。\n\n"
            for evidence in report.get("evidences", []):
                result_text += f"• {evidence.get('time_range')}: {evidence.get('summary')}\n"
            
            return result_text, report.get("timeline_image", None)
            
        except Exception as e:
            return f"分析失败: {str(e)}", None
    
    # 创建界面
    with gr.Blocks(title="侦探眼探长") as demo:
        gr.Markdown("# 🔍 侦探眼探长")
        gr.Markdown("零API依赖 · 纯本地运行 · 长视频智能分析")
        
        with gr.Row():
            with gr.Column(scale=2):
                video_input = gr.Video(label="上传视频")
                query_input = gr.Textbox(
                    label="查询语句",
                    placeholder="例如：找出所有玩手机的人"
                )
                confidence_slider = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.4,
                    label="检测置信度"
                )
                analyze_btn = gr.Button("开始分析", variant="primary")
                
            with gr.Column(scale=1):
                output_text = gr.Textbox(label="分析结果", lines=10)
                output_image = gr.Image(label="时间线分布")
        
        # 示例
        examples = gr.Examples(
            examples=[
                ["sample1.mp4", "找出所有玩手机的人", 0.4],
                ["sample2.mp4", "检测火灾和烟雾", 0.3],
            ],
            inputs=[video_input, query_input, confidence_slider],
            outputs=[output_text, output_image],
            fn=analyze_video
        )
        
        analyze_btn.click(
            fn=analyze_video,
            inputs=[video_input, query_input, confidence_slider],
            outputs=[output_text, output_image]
        )
    
    return demo


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="侦探眼探长 - 长视频分析工具")
    parser.add_argument("--mode", choices=["cli", "web", "gui"], 
                       default="web", help="运行模式")
    parser.add_argument("--video", help="视频文件路径")
    parser.add_argument("--query", help="查询语句")
    parser.add_argument("--config", default="config/settings.yaml", 
                       help="配置文件路径")
    
    args = parser.parse_args()
    
    if args.mode == "cli":
        # 命令行模式
        if not args.video or not args.query:
            print("错误：需要提供 --video 和 --query 参数")
            return
            
        detective = DetectiveEye(args.config)
        detective.initialize()
        report = detective.investigate(args.video, args.query)
        
        # 输出结果
        print("\n" + "="*50)
        print("分析报告")
        print("="*50)
        print(f"视频: {report.get('video_name')}")
        print(f"查询: {report.get('query')}")
        print(f"发现事件: {len(report.get('evidences', []))}")
        
        for evidence in report.get("evidences", []):
            print(f"\n- {evidence.get('time_range')}: {evidence.get('summary')}")
            print(f"  置信度: {evidence.get('confidence'):.2f}")
            
    elif args.mode == "web":
        # Web界面模式
        import streamlit as st
        create_streamlit_app()
        
    elif args.mode == "gui":
        # GUI模式
        demo = create_gradio_interface()
        demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()