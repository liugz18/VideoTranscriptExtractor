"""
基于WebRTC的VAD处理器实现
"""

import numpy as np
from typing import List, Optional
import logging

from ...core.vad_detector import VADDetector
from ...core.types import TimeSegment, AudioSegment, ProcessingConfig


class WebRTCVADProcessor(VADDetector):
    """基于WebRTC的VAD处理器"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.vad = None
        self._init_vad()
    
    def _init_vad(self):
        """初始化VAD"""
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(2)  # 中等敏感度
        except ImportError:
            self.logger.warning("webrtcvad未安装，使用模拟VAD")
            self.vad = None
    
    def detect_voice_segments(self, audio_data: np.ndarray, 
                            sample_rate: int) -> List[TimeSegment]:
        """检测音频中的语音区间"""
        if self.vad is None:
            # 模拟VAD检测
            return self._simulate_vad_detection(audio_data, sample_rate)
        
        try:
            # 使用WebRTC VAD
            frame_duration = 30  # 30ms帧
            frame_size = int(sample_rate * frame_duration / 1000)
            
            segments = []
            is_speech = False
            start_time = 0
            
            for i in range(0, len(audio_data) - frame_size, frame_size):
                frame = audio_data[i:i + frame_size]
                frame_bytes = frame.tobytes()
                
                try:
                    frame_is_speech = self.vad.is_speech(frame_bytes, sample_rate)
                except:
                    frame_is_speech = False
                
                current_time = i / sample_rate
                
                if frame_is_speech and not is_speech:
                    # 开始语音
                    start_time = current_time
                    is_speech = True
                elif not frame_is_speech and is_speech:
                    # 结束语音
                    end_time = current_time
                    if end_time - start_time >= self.config.vad_min_duration:
                        segments.append(TimeSegment(start=start_time, end=end_time))
                    is_speech = False
            
            # 处理最后一个语音段
            if is_speech:
                end_time = len(audio_data) / sample_rate
                if end_time - start_time >= self.config.vad_min_duration:
                    segments.append(TimeSegment(start=start_time, end=end_time))
            
            return segments
            
        except Exception as e:
            self.logger.error(f"VAD检测时发生错误: {str(e)}")
            return self._simulate_vad_detection(audio_data, sample_rate)
    
    def _simulate_vad_detection(self, audio_data: np.ndarray, 
                               sample_rate: int) -> List[TimeSegment]:
        """模拟VAD检测"""
        # 简单的能量检测
        frame_size = int(sample_rate * 0.1)  # 100ms帧
        threshold = np.mean(np.abs(audio_data)) * 2
        
        segments = []
        is_speech = False
        start_time = 0
        
        for i in range(0, len(audio_data) - frame_size, frame_size):
            frame = audio_data[i:i + frame_size]
            energy = np.mean(np.abs(frame))
            current_time = i / sample_rate
            
            if energy > threshold and not is_speech:
                start_time = current_time
                is_speech = True
            elif energy <= threshold and is_speech:
                end_time = current_time
                if end_time - start_time >= self.config.vad_min_duration:
                    segments.append(TimeSegment(start=start_time, end=end_time))
                is_speech = False
        
        # 处理最后一个语音段
        if is_speech:
            end_time = len(audio_data) / sample_rate
            if end_time - start_time >= self.config.vad_min_duration:
                segments.append(TimeSegment(start=start_time, end=end_time))
        
        return segments
    
    def detect_voice_segments_from_file(self, audio_path: str) -> List[TimeSegment]:
        """从音频文件检测语音区间"""
        try:
            # 这里需要加载音频文件
            # 暂时返回模拟结果
            return [
                TimeSegment(start=0.0, end=5.0),
                TimeSegment(start=10.0, end=15.0),
                TimeSegment(start=20.0, end=25.0)
            ]
        except Exception as e:
            self.logger.error(f"从文件检测语音区间时发生错误: {str(e)}")
            return []
    
    def detect_voice_segments_from_segments(self, 
                                          audio_segments: List[AudioSegment]) -> List[TimeSegment]:
        """从音频片段列表检测语音区间"""
        segments = []
        for segment in audio_segments:
            segment_results = self.detect_voice_segments(
                segment.audio_data, segment.sample_rate
            )
            # 调整时间戳
            for result in segment_results:
                adjusted_segment = TimeSegment(
                    start=segment.start_time + result.start,
                    end=segment.start_time + result.end
                )
                segments.append(adjusted_segment)
        
        return segments
    
    def filter_segments(self, segments: List[TimeSegment], 
                       min_duration: Optional[float] = None,
                       max_duration: Optional[float] = None) -> List[TimeSegment]:
        """过滤语音区间"""
        if min_duration is None:
            min_duration = self.config.vad_min_duration
        if max_duration is None:
            max_duration = self.config.vad_max_duration
        
        filtered = []
        for segment in segments:
            duration = segment.duration
            if min_duration <= duration <= max_duration:
                filtered.append(segment)
        
        return filtered
    
    def merge_adjacent_segments(self, segments: List[TimeSegment], 
                              gap_threshold: float = 0.5) -> List[TimeSegment]:
        """合并相邻的语音区间"""
        if not segments:
            return []
        
        # 按开始时间排序
        sorted_segments = sorted(segments, key=lambda x: x.start)
        merged = [sorted_segments[0]]
        
        for segment in sorted_segments[1:]:
            last_segment = merged[-1]
            
            # 检查是否需要合并
            if segment.start - last_segment.end <= gap_threshold:
                # 合并区间
                merged[-1] = TimeSegment(
                    start=last_segment.start,
                    end=max(last_segment.end, segment.end)
                )
            else:
                merged.append(segment)
        
        return merged
    
    def get_voice_probability(self, audio_data: np.ndarray, 
                            sample_rate: int) -> np.ndarray:
        """获取每个时间点的语音概率"""
        # 简单的能量概率计算
        frame_size = int(sample_rate * 0.1)  # 100ms帧
        probabilities = []
        
        for i in range(0, len(audio_data) - frame_size, frame_size):
            frame = audio_data[i:i + frame_size]
            energy = np.mean(np.abs(frame))
            # 简单的概率计算
            prob = min(1.0, energy / (np.mean(np.abs(audio_data)) * 2))
            probabilities.extend([prob] * frame_size)
        
        # 补齐长度
        while len(probabilities) < len(audio_data):
            probabilities.append(0.0)
        
        return np.array(probabilities[:len(audio_data)]) 