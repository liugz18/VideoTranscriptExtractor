"""
视频处理接口定义
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Generator
from pathlib import Path
import numpy as np

from .types import VideoFrame, ProcessingConfig


class VideoProcessor(ABC):
    """视频处理器接口"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
    
    @abstractmethod
    def load_video(self, video_path: str) -> bool:
        """
        加载视频文件
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            bool: 是否成功加载
        """
        pass
    
    @abstractmethod
    def get_video_info(self) -> dict:
        """
        获取视频信息
        
        Returns:
            dict: 包含视频信息的字典
                - width: 视频宽度
                - height: 视频高度
                - fps: 帧率
                - duration: 总时长（秒）
                - frame_count: 总帧数
        """
        pass
    
    @abstractmethod
    def extract_frames(self, time_segments: List[tuple]) -> List[List[VideoFrame]]:
        """
        从指定时间区间提取视频帧
        
        Args:
            time_segments: 时间区间列表，每个元素为 (start_time, end_time)
            
        Returns:
            List[List[VideoFrame]]: 每个时间区间对应的帧列表
        """
        pass
    
    @abstractmethod
    def extract_frame_at_time(self, timestamp: float) -> Optional[VideoFrame]:
        """
        在指定时间点提取单帧
        
        Args:
            timestamp: 时间戳（秒）
            
        Returns:
            VideoFrame: 视频帧数据，如果失败返回None
        """
        pass
    
    @abstractmethod
    def get_frame_generator(self, start_time: float = 0, 
                          end_time: Optional[float] = None,
                          fps: Optional[float] = None) -> Generator[VideoFrame, None, None]:
        """
        获取帧生成器
        
        Args:
            start_time: 开始时间（秒）
            end_time: 结束时间（秒），None表示到视频结束
            fps: 提取帧率，None表示使用原视频帧率
            
        Yields:
            VideoFrame: 视频帧数据
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