"""
逻辑调度与模板固化层 - 规则引擎
"""

import json
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import yaml
from pathlib import Path
from datetime import datetime
import jieba
import jieba.analyse

@dataclass
class Evidence:
    """证据"""
    id: str
    time_range: Tuple[float, float]  # 开始时间和结束时间
    summary: str
    description: str
    confidence: float
    detections: List[Dict]  # 视觉检测结果
    ocr_results: List[Dict]  # OCR识别结果
    frame_paths: List[str]  # 关键帧路径
    video_path: str
    
    def to_dict(self):
        return asdict(self)


class RuleEngine:
    """规则引擎"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.rules_config = config.get('rules', {})
        
        # 加载关键词模板
        self.keyword_templates = self._load_keyword_templates()
        
        # 初始化中文分词
        try:
            jieba.initialize()
        except:
            pass
        
        # 缓存
        self.cache = {}
    
    def _load_keyword_templates(self) -> Dict:
        """加载关键词模板"""
        template_path = self.rules_config.get(
            'template_path', 
            'config/keywords.json'
        )
        
        default_templates = {
            "安防监控": ["闯入", "异常", "打架", "跌倒", "火灾", "烟雾", "入侵"],
            "课堂分析": ["玩手机", "睡觉", "交头接耳", "讲话", "看手机", "聊天"],
            "产品演示": ["包装", "logo", "价格", "二维码", "条形码", "商标"],
            "自定义场景": []
        }
        
        if Path(template_path).exists():
            try:
                with open(template_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                print(f"⚠️  无法加载关键词模板，使用默认模板")
                return default_templates
        else:
            # 创建默认模板文件
            with open(template_path, 'w', encoding='utf-8') as f:
                json.dump(default_templates, f, ensure_ascii=False, indent=2)
            return default_templates
    
    def parse_query(self, query: str) -> List[str]:
        """解析查询语句，提取关键词"""
        # 检查是否是预设模板
        template_keywords = self._match_template(query)
        if template_keywords:
            return template_keywords
        
        # 中文分词提取关键词
        chinese_keywords = self._extract_chinese_keywords(query)
        
        # 英文关键词提取
        english_keywords = self._extract_english_keywords(query)
        
        # 合并关键词
        all_keywords = list(set(chinese_keywords + english_keywords))
        
        # 去除停用词
        stopwords = self._load_stopwords()
        filtered_keywords = [
            kw for kw in all_keywords 
            if kw.lower() not in stopwords and len(kw) > 1
        ]
        
        print(f"🔍 解析查询: '{query}' -> 关键词: {filtered_keywords}")
        return filtered_keywords
    
    def _match_template(self, query: str) -> List[str]:
        """匹配预设模板"""
        query_lower = query.lower()
        
        for category, keywords in self.keyword_templates.items():
            for keyword in keywords:
                if keyword in query_lower:
                    # 返回整个类别的关键词
                    return keywords
        
        return []
    
    def _extract_chinese_keywords(self, text: str) -> List[str]:
        """提取中文关键词"""
        keywords = []
        
        # 使用结巴分词
        try:
            words = jieba.lcut(text)
            for word in words:
                if len(word) > 1 and not word.isspace():
                    keywords.append(word)
        except:
            # 简单的中文字符提取
            chinese_chars = re.findall(r'[\u4e00-\u9fff]{2,}', text)
            keywords.extend(chinese_chars)
        
        return keywords
    
    def _extract_english_keywords(self, text: str) -> List[str]:
        """提取英文关键词"""
        keywords = []
        
        # 提取英文单词
        english_words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        
        # 转换为小写并去重
        for word in english_words:
            keywords.append(word.lower())
        
        return list(set(keywords))
    
    def _load_stopwords(self) -> List[str]:
        """加载停用词"""
        stopwords = [
            '的', '了', '在', '是', '我', '有', '和', '就',
            '不', '人', '都', '一', '一个', '上', '也', '很',
            '到', '说', '要', '去', '你', '会', '着', '没有',
            '看', '好', '自己', '这', 'the', 'and', 'a', 'an',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'
        ]
        return stopwords
    
    def match_keywords(self, segments: List, keywords: List[str], 
                      fast_mode: bool = True) -> List:
        """匹配关键词，返回嫌疑段落"""
        suspect_segments = []
        
        for segment in segments:
            # 快速模式：只检查OCR缓存
            if fast_mode:
                if hasattr(segment, 'ocr_cache'):
                    ocr_texts = segment.ocr_cache
                else:
                    # 如果没有OCR缓存，先进行OCR
                    ocr_texts = self._fast_ocr_segment(segment)
                    segment.ocr_cache = ocr_texts
                
                # 检查是否包含关键词
                if self._contains_keywords(ocr_texts, keywords):
                    suspect_segments.append(segment)
            
            # 完整模式：检查视觉和文字
            else:
                # 这里可以添加更多检查逻辑
                pass
        
        return suspect_segments
    
    def _fast_ocr_segment(self, segment) -> List[str]:
        """快速OCR扫描段落"""
        # 这里可以实现快速OCR扫描
        # 暂时返回空列表
        return []
    
    def _contains_keywords(self, texts: List[str], keywords: List[str]) -> bool:
        """检查文本是否包含关键词"""
        for text in texts:
            text_lower = text.lower()
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    return True
        return False
    
    def fuse_evidence(self, segment, detections: List, 
                     ocr_results: List, keywords: List[str]) -> Optional[Evidence]:
        """融合证据"""
        if not detections and not ocr_results:
            return None
        
        # 分析检测结果
        detection_summary = self._analyze_detections(detections, keywords)
        
        # 分析OCR结果
        ocr_summary = self._analyze_ocr(ocr_results, keywords)
        
        # 如果没有相关证据，返回None
        if not detection_summary and not ocr_summary:
            return None
        
        # 生成证据摘要
        summary = self._generate_summary(detection_summary, ocr_summary)
        
        # 计算置信度
        confidence = self._calculate_confidence(detections, ocr_results, keywords)
        
        # 创建证据对象
        evidence = Evidence(
            id=f"evidence_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{segment.start_time}",
            time_range=(segment.start_time, segment.end_time),
            summary=summary,
            description=self._generate_description(detections, ocr_results),
            confidence=confidence,
            detections=[det.__dict__ for det in detections] if detections else [],
            ocr_results=[ocr.__dict__ for ocr in ocr_results] if ocr_results else [],
            frame_paths=segment.keyframes,
            video_path=getattr(segment, 'video_path', '')
        )
        
        return evidence
    
    def _analyze_detections(self, detections: List, keywords: List[str]) -> str:
        """分析检测结果"""
        if not detections:
            return ""
        
        # 统计检测到的对象
        object_counts = {}
        for det in detections:
            label = det.label
            object_counts[label] = object_counts.get(label, 0) + 1
        
        # 生成摘要
        summary_parts = []
        for label, count in object_counts.items():
            summary_parts.append(f"{count}个{label}")
        
        return f"检测到 {'、'.join(summary_parts)}"
    
    def _analyze_ocr(self, ocr_results: List, keywords: List[str]) -> str:
        """分析OCR结果"""
        if not ocr_results:
            return ""
        
        # 提取关键词相关的文本
        relevant_texts = []
        for ocr in ocr_results:
            text = ocr.text
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    relevant_texts.append(text)
                    break
        
        if not relevant_texts:
            return ""
        
        # 去重并截断
        unique_texts = []
        for text in relevant_texts:
            if text not in unique_texts:
                if len(text) > 20:
                    unique_texts.append(text[:20] + "...")
                else:
                    unique_texts.append(text)
        
        return f"识别到文字: {'、'.join(unique_texts[:3])}"  # 最多显示3个
    
    def _generate_summary(self, detection_summary: str, 
                         ocr_summary: str) -> str:
        """生成证据摘要"""
        parts = []
        if detection_summary:
            parts.append(detection_summary)
        if ocr_summary:
            parts.append(ocr_summary)
        
        return " | ".join(parts) if parts else "未知事件"
    
    def _calculate_confidence(self, detections: List, 
                             ocr_results: List, 
                             keywords: List[str]) -> float:
        """计算证据置信度"""
        confidence = 0.0
        
        # 视觉证据权重
        if detections:
            detection_conf = sum(det.confidence for det in detections) / len(detections)
            confidence += detection_conf * 0.7  # 视觉权重70%
        
        # 文字证据权重
        if ocr_results:
            ocr_conf = sum(ocr.confidence for ocr in ocr_results) / len(ocr_results)
            confidence += ocr_conf * 0.3  # 文字权重30%
        
        # 关键词匹配加成
        keyword_bonus = 0.0
        all_texts = [ocr.text for ocr in ocr_results] if ocr_results else []
        
        for keyword in keywords:
            for text in all_texts:
                if keyword.lower() in text.lower():
                    keyword_bonus += 0.1
                    break
        
        confidence = min(1.0, confidence + min(keyword_bonus, 0.2))
        
        return confidence
    
    def _generate_description(self, detections: List, 
                            ocr_results: List) -> str:
        """生成详细描述"""
        descriptions = []
        
        # 视觉检测描述
        if detections:
            obj_counts = {}
            for det in detections:
                label = det.label
                obj_counts[label] = obj_counts.get(label, 0) + 1
            
            obj_desc = []
            for label, count in obj_counts.items():
                obj_desc.append(f"{count}个{label}")
            
            if obj_desc:
                descriptions.append(f"视觉检测: 发现{', '.join(obj_desc)}")
        
        # OCR结果描述
        if ocr_results:
            unique_texts = []
            for ocr in ocr_results[:3]:  # 最多显示3个
                text = ocr.text
                if len(text) > 15:
                    text = text[:15] + "..."
                if text not in unique_texts:
                    unique_texts.append(text)
            
            if unique_texts:
                descriptions.append(f"文字识别: {', '.join(unique_texts)}")
        
        return " | ".join(descriptions) if descriptions else "无详细描述"
    
    def save_rules(self, category: str, keywords: List[str]):
        """保存规则到模板"""
        if category in self.keyword_templates:
            self.keyword_templates[category] = keywords
            
            # 保存到文件
            template_path = self.rules_config.get(
                'template_path', 
                'config/keywords.json'
            )
            
            with open(template_path, 'w', encoding='utf-8') as f:
                json.dump(self.keyword_templates, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 已保存规则: {category} -> {keywords}")
    
    def load_custom_rules(self, rule_file: str):
        """加载自定义规则"""
        if Path(rule_file).exists():
            try:
                with open(rule_file, 'r', encoding='utf-8') as f:
                    custom_rules = json.load(f)
                
                # 合并规则
                for category, keywords in custom_rules.items():
                    if category in self.keyword_templates:
                        self.keyword_templates[category].extend(keywords)
                    else:
                        self.keyword_templates[category] = keywords
                
                print(f"✅ 已加载自定义规则: {rule_file}")
                
            except Exception as e:
                print(f"❌ 加载自定义规则失败: {e}")