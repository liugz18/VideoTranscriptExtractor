"""
结果融合处理接口定义
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np

from .types import OCRResult, ASRResult, TranscriptionSegment, ProcessingConfig


class FusionProcessor(ABC):
    """结果融合处理器接口"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
    
    @abstractmethod
    def fuse_results(self, ocr_results: List[OCRResult], 
                    asr_results: List[ASRResult]) -> List[TranscriptionSegment]:
        """
        融合OCR和ASR结果
        
        Args:
            ocr_results: OCR识别结果列表
            asr_results: ASR识别结果列表
            
        Returns:
            List[TranscriptionSegment]: 融合后的转录片段列表
        """
        pass
    
    @abstractmethod
    def align_results(self, ocr_results: List[OCRResult], 
                     asr_results: List[ASRResult]) -> List[Dict[str, Any]]:
        """
        对齐OCR和ASR结果
        
        Args:
            ocr_results: OCR识别结果列表
            asr_results: ASR识别结果列表
            
        Returns:
            List[Dict[str, Any]]: 对齐后的结果列表
        """
        pass
    
    @abstractmethod
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            float: 相似度分数 (0-1)
        """
        pass
    
    @abstractmethod
    def merge_similar_texts(self, texts: List[str], 
                          similarity_threshold: float = 0.8) -> List[str]:
        """
        合并相似的文本
        
        Args:
            texts: 文本列表
            similarity_threshold: 相似度阈值
            
        Returns:
            List[str]: 合并后的文本列表
        """
        pass
    
    @abstractmethod
    def select_best_result(self, ocr_result: OCRResult, 
                          asr_result: ASRResult) -> Dict[str, Any]:
        """
        选择最佳结果
        
        Args:
            ocr_result: OCR结果
            asr_result: ASR结果
            
        Returns:
            Dict[str, Any]: 包含最佳结果和来源的字典
        """
        pass
    
    @abstractmethod
    def interpolate_missing_segments(self, segments: List[TranscriptionSegment], 
                                   total_duration: float) -> List[TranscriptionSegment]:
        """
        插值缺失的转录片段
        
        Args:
            segments: 转录片段列表
            total_duration: 总时长
            
        Returns:
            List[TranscriptionSegment]: 插值后的片段列表
        """
        pass
    
    @abstractmethod
    def smooth_segments(self, segments: List[TranscriptionSegment], 
                       window_size: float = 1.0) -> List[TranscriptionSegment]:
        """
        平滑转录片段
        
        Args:
            segments: 转录片段列表
            window_size: 平滑窗口大小（秒）
            
        Returns:
            List[TranscriptionSegment]: 平滑后的片段列表
        """
        pass
    
    def process_fusion(self, ocr_results: List[OCRResult], 
                      asr_results: List[ASRResult]) -> List[TranscriptionSegment]:
        """
        完整的融合处理流程
        
        Args:
            ocr_results: OCR识别结果列表
            asr_results: ASR识别结果列表
            
        Returns:
            List[TranscriptionSegment]: 处理后的转录片段列表
        """
        # 对齐结果
        aligned_results = self.align_results(ocr_results, asr_results)
        
        # 融合结果
        segments = self.fuse_results(ocr_results, asr_results)
        
        # 平滑片段
        segments = self.smooth_segments(segments)
        
        return segments
    
    @abstractmethod
    def set_fusion_weights(self, ocr_weight: float, asr_weight: float):
        """
        设置融合权重
        
        Args:
            ocr_weight: OCR权重
            asr_weight: ASR权重
        """
        pass
    
    @abstractmethod
    def get_fusion_strategy(self) -> str:
        """
        获取融合策略
        
        Returns:
            str: 融合策略名称
        """
        pass
    
    @abstractmethod
    def set_fusion_strategy(self, strategy: str):
        """
        设置融合策略
        
        Args:
            strategy: 融合策略名称
        """
        pass 