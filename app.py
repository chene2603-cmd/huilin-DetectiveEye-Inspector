#!/usr/bin/env python3
"""
DetectiveEye Inspector - 视频行为智能分析系统（主入口）
零API依赖 · 纯本地运行
"""

import argparse
import os
import sys
import time
from pathlib import Path

# 确保项目根目录在 sys.path 中，方便导入 core 模块
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from core.utils import (
    load_config, save_config, cleanup_temp_files,
    check_system_requirements, print_system_info, setup_logging
)
from core.video_processor import VideoProcessor
from core.detector import YOLODetector
from core.ocr_engine import OCREngine
from core.rule_engine import RuleEngine


def parse_args():
    parser = argparse.ArgumentParser(
        description="DetectiveEye Inspector - 视频行为智能分析系统"
    )
    parser.add_argument("--video", "-v", required=True, help="视频文件路径")
    parser.add_argument("--query", "-q", required=True, help="分析查询，例如：'检测玩手机行为'")
    parser.add_argument("--output", "-o", default=None, help="输出目录（默认自动生成）")
    parser.add_argument("--config", "-c", default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--fast", action="store_true", help="快速模式（优先使用缓存）")
    parser.add_argument("--no-cache", action="store_true", help="禁用缓存")
    parser.add_argument("--check-system", action="store_true", help="检查系统环境并退出")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    return parser.parse_args()


def main():
    args = parse_args()

    # 设置日志
    logger = setup_logging()
    if args.debug:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("调试模式已启用")

    # 如果需要检查系统
    if args.check_system:
        print_system_info()
        reqs, issues = check_system_requirements()
        if issues:
            print("\n⚠️  发现问题：")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\n✅ 所有系统检查通过")
        return

    # 检查视频文件
    if not os.path.exists(args.video):
        logger.error(f"视频文件不存在: {args.video}")
        sys.exit(1)

    # 加载配置
    if not os.path.exists(args.config):
        logger.warning(f"配置文件不存在: {args.config}，将创建默认配置")
    config = load_config(args.config)

    # 检查系统依赖（首次运行可能会下载模型）
    logger.info("检查系统依赖...")
    reqs, issues = check_system_requirements()
    if issues:
        logger.warning("发现以下问题：")
        for issue in issues:
            logger.warning(f"  {issue}")
        if not args.fast:
            logger.error("关键依赖缺失，无法继续。请解决后再试。")
            sys.exit(1)

    # 初始化组件
    logger.info("初始化分析引擎...")
    detector = YOLODetector(
        model_path=config["models"]["yolo"]["model_path"],
        confidence=config["models"]["yolo"]["confidence"],
        device="cuda" if config.get("gpu", True) else "cpu"
    )
    ocr = OCREngine(
        languages=config["models"]["ocr"]["languages"],
        use_gpu=config["models"]["ocr"]["gpu"]
    )
    video_processor = VideoProcessor(
        preview_fps=config["video"]["preview_fps"],
        detail_fps=config["video"]["detail_fps"],
        scene_threshold=config["video"]["scene_threshold"]
    )
    rule_engine = RuleEngine(
        template_path=config["rules"]["template_path"]
    )

    # 分析流程
    start_time = time.time()
    logger.info(f"开始分析: {args.video}")
    logger.info(f"查询意图: {args.query}")

    try:
        # 视频处理
        frames_info = video_processor.extract_frames(
            args.video,
            use_cache=not args.no_cache
        )
        logger.info(f"提取到 {len(frames_info)} 个关键帧")

        # 视觉检测
        detections = detector.detect_batch(
            [f["frame"] for f in frames_info],
            batch_size=config["models"]["yolo"]["batch_size"]
        )

        # OCR识别
        ocr_results = ocr.recognize_batch(
            [f["frame"] for f in frames_info],
            batch_size=config["models"]["ocr"]["batch_size"]
        )

        # 规则匹配
        evidences = rule_engine.analyze(
            query=args.query,
            frames_info=frames_info,
            detections=detections,
            ocr_results=ocr_results,
            fast_mode=args.fast
        )
        logger.info(f"发现 {len(evidences)} 个相关事件")

        # 生成输出
        from core.utils import create_output_directory
        output_dir = args.output or create_output_directory(args.video)
        logger.info(f"输出目录: {output_dir}")

        # 保存证据帧
        for ev in evidences:
            ev.save_frames(output_dir)

        # 生成报告
        from core.utils import generate_report, save_report_markdown, save_report_json
        report = generate_report(evidences, args.video, args.query, config)
        save_report_markdown(report, output_dir)
        save_report_json(report, output_dir)

        # 可视化（可选）
        from core.utils import create_visualization
        create_visualization(report, output_dir)

        # 清理临时文件
        if config["system"].get("temp_cleanup", True):
            cleanup_temp_files()

        elapsed = time.time() - start_time
        logger.info(f"分析完成，耗时 {elapsed:.1f} 秒")
        logger.info(f"报告已保存至: {output_dir}")

    except Exception as e:
        logger.exception(f"分析过程出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()