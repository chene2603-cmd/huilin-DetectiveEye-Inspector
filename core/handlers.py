"""
行为匹配处理器（责任链模式）
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class BaseHandler(ABC):
    """行为匹配处理器基类"""

    def __init__(self):
        self._next: Optional[BaseHandler] = None

    def set_next(self, handler: 'BaseHandler') -> 'BaseHandler':
        self._next = handler
        return handler

    @abstractmethod
    def can_handle(self, query: str) -> bool:
        ...

    @abstractmethod
    def handle(self, query: str, detections: List[Dict],
               ocr_results: List[Dict]) -> List[Dict]:
        ...

    def process(self, query: str, detections: List[Dict],
                ocr_results: List[Dict]) -> Optional[List[Dict]]:
        if self.can_handle(query):
            return self.handle(query, detections, ocr_results)
        elif self._next:
            return self._next.process(query, detections, ocr_results)
        return None


# ========== 具体行为处理器 ==========

class PhoneUsageHandler(BaseHandler):
    """匹配'玩手机'行为"""
    KEYWORDS = ["玩手机", "打电话", "用手机", "看手机"]

    def can_handle(self, query: str) -> bool:
        return any(kw in query for kw in self.KEYWORDS)

    def handle(self, query: str, detections: List[Dict],
               ocr_results: List[Dict]) -> List[Dict]:
        evidences = []
        for det in detections:
            if det.get("label") == "cell phone":
                evidences.append({
                    "type": "phone_usage",
                    "detection": det,
                    "action": "使用手机"
                })
        return evidences


class SleepingHandler(BaseHandler):
    """匹配'睡觉'行为"""
    KEYWORDS = ["睡觉", "趴着", "打瞌睡", "打盹"]

    def can_handle(self, query: str) -> bool:
        return any(kw in query for kw in self.KEYWORDS)

    def handle(self, query: str, detections: List[Dict],
               ocr_results: List[Dict]) -> List[Dict]:
        evidences = []
        for det in detections:
            if det.get("label") == "person":
                evidences.append({
                    "type": "sleeping",
                    "detection": det,
                    "action": "疑似睡觉"
                })
        return evidences


class GeneralKeywordHandler(BaseHandler):
    """兜底处理器：基于 OCR 文本模糊匹配"""

    def can_handle(self, query: str) -> bool:
        return True

    def handle(self, query: str, detections: List[Dict],
               ocr_results: List[Dict]) -> List[Dict]:
        evidences = []
        for ocr in ocr_results:
            if query in ocr.get("text", ""):
                evidences.append({
                    "type": "keyword_match",
                    "ocr_result": ocr,
                    "action": f"匹配关键词: {query}"
                })
        return evidences