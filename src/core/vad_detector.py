"""
VAD (Voice Activity Detection) 检测接口定义
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from .types import TimeSegment, AudioSegment, ProcessingConfig


class VADDetector(ABC):
    """VAD检测器接口"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
    
    @abstractmethod
    def detect_voice_segments(self, audio_data: np.ndarray, 
                            sample_rate: int) -> List[TimeSegment]:
        """
        检测音频中的语音区间
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            
        Returns:
            List[TimeSegment]: 语音区间列表，按时间排序
        """
        pass
    
    @abstractmethod
    def detect_voice_segments_from_file(self, audio_path: str) -> List[TimeSegment]:
        """
        从音频文件检测语音区间
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            List[TimeSegment]: 语音区间列表
        """
        pass
    
    @abstractmethod
    def detect_voice_segments_from_segments(self, 
                                          audio_segments: List[AudioSegment]) -> List[TimeSegment]:
        """
        从音频片段列表检测语音区间
        
        Args:
            audio_segments: 音频片段列表
            
        Returns:
            List[TimeSegment]: 语音区间列表
        """
        pass
    
    @abstractmethod
    def filter_segments(self, segments: List[TimeSegment], 
                       min_duration: Optional[float] = None,
                       max_duration: Optional[float] = None) -> List[TimeSegment]:
        """
        过滤语音区间
        
        Args:
            segments: 原始语音区间列表
            min_duration: 最小持续时间（秒）
            max_duration: 最大持续时间（秒）
            
        Returns:
            List[TimeSegment]: 过滤后的语音区间列表
        """
        pass
    
    @abstractmethod
    def merge_adjacent_segments(self, segments: List[TimeSegment], 
                              gap_threshold: float = 0.5) -> List[TimeSegment]:
        """
        合并相邻的语音区间
        
        Args:
            segments: 语音区间列表
            gap_threshold: 间隔阈值（秒），小于此值的区间将被合并
            
        Returns:
            List[TimeSegment]: 合并后的语音区间列表
        """
        pass
    
    @abstractmethod
    def get_voice_probability(self, audio_data: np.ndarray, 
                            sample_rate: int) -> np.ndarray:
        """
        获取每个时间点的语音概率
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            
        Returns:
            np.ndarray: 语音概率数组，长度与音频数据对应
        """
        pass
    
    def process_audio(self, audio_data: np.ndarray, sample_rate: int) -> List[TimeSegment]:
        """
        完整的音频处理流程
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            
        Returns:
            List[TimeSegment]: 处理后的语音区间列表
        """
        # 检测语音区间
        segments = self.detect_voice_segments(audio_data, sample_rate)
        
        # 过滤短区间
        segments = self.filter_segments(
            segments, 
            min_duration=self.config.vad_min_duration,
            max_duration=self.config.vad_max_duration
        )
        
        # 合并相邻区间
        segments = self.merge_adjacent_segments(segments)
        
        return segments 