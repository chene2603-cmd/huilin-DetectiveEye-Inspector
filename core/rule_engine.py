"""
规则引擎模块
结合视觉检测、OCR 文字与用户查询意图，通过责任链模式匹配行为证据
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from core.handlers import (
    BaseHandler,
    PhoneUsageHandler,
    SleepingHandler,
    GeneralKeywordHandler
)


class Evidence:
    """证据对象，统一封装分析结果"""
    def __init__(self, evidence_id: str, time_range, summary: str,
                 description: str, confidence: float,
                 detections: List, ocr_results: List,
                 frame_paths: List[str]):
        self.id = evidence_id
        self.time_range = time_range          # (start, end) 秒
        self.summary = summary
        self.description = description
        self.confidence = confidence
        self.detections = detections
        self.ocr_results = ocr_results
        self.frame_paths = frame_paths


class RuleEngine:
    def __init__(self, template_path: str = "config/keywords.json"):
        self.template_path = template_path
        self.keywords = self._load_templates()

        # 装配责任链
        self.handler_chain = (
            PhoneUsageHandler()
            .set_next(SleepingHandler())
            .set_next(GeneralKeywordHandler())
        )

    def _load_templates(self) -> Dict:
        """从配置文件加载关键词模板"""
        path = Path(self.template_path)
        if not path.exists():
            print(f"⚠️  未找到关键词模板: {path}，使用默认配置")
            return {"default": []}
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def parse_query(self, query: str) -> List[str]:
        """
        解析用户查询，返回意图关键词列表
        优先使用责任链处理，若未命中则回退到配置文件匹配
        """
        # 责任链尝试
        chain_result = self.handler_chain.process(query, detections=[], ocr_results=[])
        if chain_result:
            keywords = []
            for item in chain_result:
                keywords.extend(item.get("action", "").split())
            return keywords

        # 回退：基于配置文件的简单关键词匹配
        all_keywords = []
        for category, words in self.keywords.items():
            for word in words:
                if word in query:
                    all_keywords.append(word)
        return all_keywords if all_keywords else [query]

    def analyze(
        self,
        query: str,
        frames_info: List[Dict],
        detections: List[List[Dict]],
        ocr_results: List[List[Dict]],
        fast_mode: bool = False
    ) -> List[Evidence]:
        """
        主分析入口
        Args:
            query: 用户查询文本
            frames_info: VideoProcessor 返回的帧信息列表
            detections: 各帧的检测结果列表
            ocr_results: 各帧的 OCR 结果列表
            fast_mode: 是否快速模式（会影响证据粒度）
        Returns:
            Evidence 对象列表
        """
        keywords = self.parse_query(query)
        print(f"🔑 解析关键词: {keywords}")

        evidences = []
        ev_id = 0

        # 遍历所有帧，收集潜在证据片段
        for frame_idx, (frame_info, dets, ocrs) in enumerate(
            zip(frames_info, detections, ocr_results)
        ):
            # 通过责任链生成证据片段
            segment = self.handler_chain.process(query, dets, ocrs)
            if segment:
                ev_id += 1
                # 提取时间范围（当前帧为中心，前后各1秒）
                start = max(0, frame_info["timestamp"] - 1)
                end = frame_info["timestamp"] + 1
                evidence = Evidence(
                    evidence_id=f"ev_{ev_id:04d}",
                    time_range=(start, end),
                    summary=query,
                    description=f"在 {frame_info['timestamp']:.1f}秒 附近发现匹配行为",
                    confidence=0.85 if fast_mode else 0.9,  # 简单示例
                    detections=dets,
                    ocr_results=ocrs,
                    frame_paths=[]
                )
                evidences.append(evidence)

        # 合并相邻的短证据
        evidences = self._merge_adjacent_evidences(evidences)
        return evidences

    def _merge_adjacent_evidences(self, evidences: List[Evidence],
                                  max_gap: float = 3.0) -> List[Evidence]:
        """合并时间相邻的同类证据"""
        if len(evidences) < 2:
            return evidences
        merged = [evidences[0]]
        for ev in evidences[1:]:
            last = merged[-1]
            if ev.time_range[0] - last.time_range[1] <= max_gap:
                # 合并
                merged[-1] = Evidence(
                    evidence_id=last.id,
                    time_range=(last.time_range[0], ev.time_range[1]),
                    summary=last.summary,
                    description=f"{last.description} 至 {ev.time_range[1]:.1f}秒",
                    confidence=max(last.confidence, ev.confidence),
                    detections=last.detections + ev.detections,
                    ocr_results=last.ocr_results + ev.ocr_results,
                    frame_paths=list(set(last.frame_paths + ev.frame_paths))
                )
            else:
                merged.append(ev)
        return merged