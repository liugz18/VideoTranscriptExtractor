"""
基于OpenCV的视频处理器实现
"""

import cv2
import numpy as np
from typing import List, Optional, Generator
import logging

from ...core.video_processor import VideoProcessor
from ...core.types import VideoFrame, ProcessingConfig


class OpenCVVideoProcessor(VideoProcessor):
    """基于OpenCV的视频处理器"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        super().__init__(config)
        self.cap = None
        self.video_info = {}
        self.logger = logging.getLogger(__name__)
    
    def load_video(self, video_path: str) -> bool:
        """加载视频文件"""
        try:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                self.logger.error(f"无法打开视频文件: {video_path}")
                return False
            
            # 获取视频信息
            self.video_info = {
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.cap.get(cv2.CAP_PROP_FPS),
                'duration': self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            }
            
            self.logger.info(f"成功加载视频: {video_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"加载视频时发生错误: {str(e)}")
            return False
    
    def get_video_info(self) -> dict:
        """获取视频信息"""
        return self.video_info.copy()
    
    def extract_frames(self, time_segments: List[tuple]) -> List[List[VideoFrame]]:
        """从指定时间区间提取视频帧"""
        if not self.cap:
            raise RuntimeError("视频未加载")
        
        results = []
        for start_time, end_time in time_segments:
            frames = []
            start_frame = int(start_time * self.video_info['fps'])
            end_frame = int(end_time * self.video_info['fps'])
            
            # 设置起始帧
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            for frame_idx in range(start_frame, end_frame):
                ret, frame = self.cap.read()
                if ret:
                    # 转换BGR到RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    timestamp = frame_idx / self.video_info['fps']
                    
                    video_frame = VideoFrame(
                        timestamp=timestamp,
                        frame_data=frame_rgb,
                        frame_number=frame_idx
                    )
                    frames.append(video_frame)
                else:
                    break
            
            results.append(frames)
        
        return results
    
    def extract_frame_at_time(self, timestamp: float) -> Optional[VideoFrame]:
        """在指定时间点提取单帧"""
        if not self.cap:
            return None
        
        frame_idx = int(timestamp * self.video_info['fps'])
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return VideoFrame(
                timestamp=timestamp,
                frame_data=frame_rgb,
                frame_number=frame_idx
            )
        
        return None
    
    def get_frame_generator(self, start_time: float = 0, 
                          end_time: Optional[float] = None,
                          fps: Optional[float] = None) -> Generator[VideoFrame, None, None]:
        """获取帧生成器"""
        if not self.cap:
            return
        
        start_frame = int(start_time * self.video_info['fps'])
        if end_time is None:
            end_frame = self.video_info['frame_count']
        else:
            end_frame = int(end_time * self.video_info['fps'])
        
        # 设置起始帧
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_interval = 1
        if fps and fps < self.video_info['fps']:
            frame_interval = int(self.video_info['fps'] / fps)
        
        frame_idx = start_frame
        while frame_idx < end_frame:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if (frame_idx - start_frame) % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp = frame_idx / self.video_info['fps']
                
                yield VideoFrame(
                    timestamp=timestamp,
                    frame_data=frame_rgb,
                    frame_number=frame_idx
                )
            
            frame_idx += 1
    
    def close(self):
        """释放资源"""
        if self.cap:
            self.cap.release()
            self.cap = None 