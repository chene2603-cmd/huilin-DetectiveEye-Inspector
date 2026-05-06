from abc import ABC, abstractmethod
from typing import List, Dict, Optional

class BaseHandler(ABC):
    """行为匹配处理器基类"""
    
    def __init__(self):
        self._next = None

    def set_next(self, handler: 'BaseHandler') -> 'BaseHandler':
        """设置下一个处理器，形成责任链"""
        self._next = handler
        return handler  # 允许链式调用

    @abstractmethod
    def can_handle(self, query: str) -> bool:
        """判断能否处理该查询"""
        pass

    @abstractmethod
    def handle(self, query: str, detections: List[Dict], ocr_results: List[Dict]) -> List[Dict]:
        """具体的处理逻辑，返回证据列表"""
        pass

    def process(self, query: str, detections: List[Dict], ocr_results: List[Dict]) -> Optional[List[Dict]]:
        """
        责任链执行入口：
        如果能处理则处理，否则交给下一个处理器
        """
        if self.can_handle(query):
            return self.handle(query, detections, ocr_results)
        elif self._next:
            return self._next.process(query, detections, ocr_results)
        return None