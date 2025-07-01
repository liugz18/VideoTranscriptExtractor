"""
OCR (Optical Character Recognition) 处理接口定义
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np

from .types import VideoFrame, OCRResult, ProcessingConfig


class OCRProcessor(ABC):
    """OCR处理器接口"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
    
    @abstractmethod
    def recognize_text(self, image: np.ndarray) -> List[OCRResult]:
        """
        识别图像中的文本
        
        Args:
            image: 输入图像
            
        Returns:
            List[OCRResult]: OCR识别结果列表
        """
        pass
    
    @abstractmethod
    def recognize_text_from_frames(self, frames: List[VideoFrame]) -> List[OCRResult]:
        """
        从视频帧列表识别文本
        
        Args:
            frames: 视频帧列表
            
        Returns:
            List[OCRResult]: OCR识别结果列表
        """
        pass
    
    @abstractmethod
    def recognize_text_from_file(self, image_path: str) -> List[OCRResult]:
        """
        从图像文件识别文本
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            List[OCRResult]: OCR识别结果列表
        """
        pass
    
    @abstractmethod
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理
        
        Args:
            image: 原始图像
            
        Returns:
            np.ndarray: 预处理后的图像
        """
        pass
    
    @abstractmethod
    def detect_text_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        检测文本区域
        
        Args:
            image: 输入图像
            
        Returns:
            List[Dict[str, Any]]: 文本区域列表，每个元素包含位置信息
        """
        pass
    
    @abstractmethod
    def filter_results(self, results: List[OCRResult], 
                      confidence_threshold: Optional[float] = None) -> List[OCRResult]:
        """
        过滤OCR结果
        
        Args:
            results: OCR结果列表
            confidence_threshold: 置信度阈值
            
        Returns:
            List[OCRResult]: 过滤后的结果列表
        """
        pass
    
    @abstractmethod
    def merge_results(self, results: List[OCRResult], 
                     time_threshold: float = 1.0) -> List[OCRResult]:
        """
        合并时间相近的OCR结果
        
        Args:
            results: OCR结果列表
            time_threshold: 时间阈值（秒）
            
        Returns:
            List[OCRResult]: 合并后的结果列表
        """
        pass
    
    def process_frames(self, frames: List[VideoFrame]) -> List[OCRResult]:
        """
        完整的帧处理流程
        
        Args:
            frames: 视频帧列表
            
        Returns:
            List[OCRResult]: 处理后的OCR结果列表
        """
        # 识别文本
        results = self.recognize_text_from_frames(frames)
        
        # 过滤低置信度结果
        results = self.filter_results(
            results, 
            confidence_threshold=self.config.ocr_confidence_threshold
        )
        
        # 合并相近时间的结果
        results = self.merge_results(results)
        
        return results
    
    @abstractmethod
    def set_languages(self, languages: List[str]):
        """
        设置识别语言
        
        Args:
            languages: 语言代码列表，如 ['en', 'zh']
        """
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """
        获取支持的语言列表
        
        Returns:
            List[str]: 支持的语言代码列表
        """
        pass 