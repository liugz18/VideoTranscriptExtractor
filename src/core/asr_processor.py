"""
ASR (Automatic Speech Recognition) 处理接口定义
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np

from .types import AudioSegment, ASRResult, ProcessingConfig


class ASRProcessor(ABC):
    """ASR处理器接口"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
    
    @abstractmethod
    def transcribe_audio(self, audio_data: np.ndarray, 
                        sample_rate: int) -> List[ASRResult]:
        """
        转录音频数据
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            
        Returns:
            List[ASRResult]: ASR识别结果列表
        """
        pass
    
    @abstractmethod
    def transcribe_audio_file(self, audio_path: str) -> List[ASRResult]:
        """
        转录音频文件
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            List[ASRResult]: ASR识别结果列表
        """
        pass
    
    @abstractmethod
    def transcribe_audio_segments(self, audio_segments: List[AudioSegment]) -> List[ASRResult]:
        """
        转录音频片段列表
        
        Args:
            audio_segments: 音频片段列表
            
        Returns:
            List[ASRResult]: ASR识别结果列表
        """
        pass
    
    @abstractmethod
    def transcribe_with_timestamps(self, audio_data: np.ndarray, 
                                 sample_rate: int) -> List[ASRResult]:
        """
        带时间戳的转录
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            
        Returns:
            List[ASRResult]: 包含时间戳的ASR结果列表
        """
        pass
    
    @abstractmethod
    def detect_language(self, audio_data: np.ndarray, 
                       sample_rate: int) -> str:
        """
        检测音频语言
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            
        Returns:
            str: 语言代码
        """
        pass
    
    @abstractmethod
    def set_language(self, language: str):
        """
        设置识别语言
        
        Args:
            language: 语言代码，如 'en', 'zh', 'auto'
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
    
    @abstractmethod
    def preprocess_audio(self, audio_data: np.ndarray, 
                        sample_rate: int) -> np.ndarray:
        """
        音频预处理
        
        Args:
            audio_data: 原始音频数据
            sample_rate: 采样率
            
        Returns:
            np.ndarray: 预处理后的音频数据
        """
        pass
    
    @abstractmethod
    def filter_results(self, results: List[ASRResult], 
                      confidence_threshold: Optional[float] = None) -> List[ASRResult]:
        """
        过滤ASR结果
        
        Args:
            results: ASR结果列表
            confidence_threshold: 置信度阈值
            
        Returns:
            List[ASRResult]: 过滤后的结果列表
        """
        pass
    
    def process_audio_segments(self, audio_segments: List[AudioSegment]) -> List[ASRResult]:
        """
        完整的音频片段处理流程
        
        Args:
            audio_segments: 音频片段列表
            
        Returns:
            List[ASRResult]: 处理后的ASR结果列表
        """
        # 转录音频片段
        results = self.transcribe_audio_segments(audio_segments)
        
        # 过滤低置信度结果
        results = self.filter_results(results)
        
        return results
    
    @abstractmethod
    def load_model(self, model_name: Optional[str] = None):
        """
        加载ASR模型
        
        Args:
            model_name: 模型名称，None使用默认模型
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息字典
        """
        pass 