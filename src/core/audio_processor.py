"""
音频处理接口定义
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np

from .types import AudioSegment, ProcessingConfig


class AudioProcessor(ABC):
    """音频处理器接口"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
    
    @abstractmethod
    def extract_audio_from_video(self, video_path: str, 
                               output_path: Optional[str] = None) -> str:
        """
        从视频中提取音频
        
        Args:
            video_path: 视频文件路径
            output_path: 音频输出路径，None表示使用临时文件
            
        Returns:
            str: 音频文件路径
        """
        pass
    
    @abstractmethod
    def load_audio(self, audio_path: str) -> bool:
        """
        加载音频文件
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            bool: 是否成功加载
        """
        pass
    
    @abstractmethod
    def get_audio_info(self) -> dict:
        """
        获取音频信息
        
        Returns:
            dict: 包含音频信息的字典
                - sample_rate: 采样率
                - duration: 总时长（秒）
                - channels: 声道数
                - format: 音频格式
        """
        pass
    
    @abstractmethod
    def extract_audio_segments(self, time_segments: List[Tuple[float, float]]) -> List[AudioSegment]:
        """
        提取指定时间区间的音频片段
        
        Args:
            time_segments: 时间区间列表，每个元素为 (start_time, end_time)
            
        Returns:
            List[AudioSegment]: 音频片段列表
        """
        pass
    
    @abstractmethod
    def resample_audio(self, audio_data: np.ndarray, 
                      original_rate: int, target_rate: int) -> np.ndarray:
        """
        重采样音频
        
        Args:
            audio_data: 音频数据
            original_rate: 原始采样率
            target_rate: 目标采样率
            
        Returns:
            np.ndarray: 重采样后的音频数据
        """
        pass
    
    @abstractmethod
    def normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        音频归一化
        
        Args:
            audio_data: 音频数据
            
        Returns:
            np.ndarray: 归一化后的音频数据
        """
        pass
    
    @abstractmethod
    def save_audio(self, audio_data: np.ndarray, output_path: str, 
                  sample_rate: int, format: str = "wav") -> bool:
        """
        保存音频文件
        
        Args:
            audio_data: 音频数据
            output_path: 输出路径
            sample_rate: 采样率
            format: 音频格式
            
        Returns:
            bool: 是否成功保存
        """
        pass
    
    @abstractmethod
    def close(self):
        """释放资源"""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 