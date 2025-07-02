"""
结果融合处理接口定义
"""

from abc import ABC, abstractmethod
from typing import Any
from .types import TranscriptionSegment, TranscriptionResult


class FusionProcessor(ABC):
    """结果融合处理器接口"""
    
    @abstractmethod
    def fuse_segment(self, seg1: TranscriptionSegment, seg2: TranscriptionSegment) -> TranscriptionSegment:
        """
        融合两个TranscriptionSegment，返回一个新的TranscriptionSegment
        """
        pass
    
    @abstractmethod
    def fuse_results(self, result1: TranscriptionResult, result2: TranscriptionResult) -> TranscriptionResult:
        """
        融合两个TranscriptionResult，返回一个新的TranscriptionResult
        """
        pass