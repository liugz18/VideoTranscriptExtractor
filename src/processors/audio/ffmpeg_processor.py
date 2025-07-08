"""
基于FFmpeg的音频处理器实现
"""

import subprocess
import tempfile
import os
from typing import List, Optional, Tuple
import numpy as np
import logging

from ...core.audio_processor import AudioProcessor
from ...core.types import AudioSegment, ProcessingConfig
from pydub import AudioSegment as PydubAudioSegment

class FFmpegAudioProcessor(AudioProcessor):
    """基于FFmpeg的音频处理器"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
    
    def extract_audio_from_video(self, video_path: str, 
                               output_path: Optional[str] = None) -> Optional[str]:
        """从视频中提取音频"""
        if output_path is None:
            # 创建临时文件
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            output_path = temp_file.name
            temp_file.close()
        
        try:
            # 使用FFmpeg提取音频
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn',  # 不包含视频
                '-acodec', 'pcm_s16le',  # 16位PCM编码
                '-ar', '16000',  # 16kHz采样率
                '-ac', '1',  # 单声道
                '-y',  # 覆盖输出文件
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"FFmpeg提取音频失败: {result.stderr}")
                return None
            
            self.logger.info(f"音频提取成功: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"提取音频时发生错误: {str(e)}")
            return None
    
    def load_audio(self, audio_path: str) -> bool:
        """加载音频文件"""
        # FFmpeg处理器不需要预加载音频
        return os.path.exists(audio_path)
    
    def get_audio_info(self) -> dict:
        """获取音频信息"""
        # 这里可以实现获取音频信息的逻辑
        return {
            'sample_rate': 16000,
            'duration': 0.0,
            'channels': 1,
            'format': 'wav'
        }
    
    def extract_audio_segments(self, time_segments: List[Tuple[float, float]],audio_path: str) -> List[AudioSegment]:
        """提取指定时间区间的音频片段"""
        segments = []
        print(f"audio_path::{audio_path}") 
        for start_time, end_time in time_segments:
            # 创建临时文件
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_path = temp_file.name
            temp_file.close()          
            try:
                # 使用FFmpeg提取音频片段
                cmd = [
                    'ffmpeg', '-i', audio_path,  # 需要先设置输入文件
                    '-ss', str(start_time),
                    '-t', str(end_time - start_time),
                    '-acodec', 'pcm_s16le',
                    '-ar', '16000',
                    '-ac', '1',
                    '-y',
                    temp_path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    self.logger.error(f"FFmpeg提取音频失败: {result.stderr}")
                    continue  # 跳过这个片段
                # 这里需要实际的音频文件路径
                # 暂时返回空的音频数据
               # audio_data = np.zeros(int((end_time - start_time) * 16000), dtype=np.int16)
                # 用pydub读临时文件
                pydub_seg = PydubAudioSegment.from_wav(temp_path)
                audio_data = np.array(pydub_seg.get_array_of_samples(), dtype=np.int16)
                segment = AudioSegment(
                    start_time=start_time,
                    end_time=end_time,
                    audio_data=audio_data,
                    path=temp_path,
                    sample_rate=16000
                )
                segments.append(segment)
                
            except Exception as e:
                self.logger.error(f"提取音频片段时发生错误: {str(e)}")
            finally:
                # 清理临时文件
                # if os.path.exists(temp_path):
                #     os.unlink(temp_path)
                pass
        
        return segments
    
    def resample_audio(self, audio_data: np.ndarray, 
                      original_rate: int, target_rate: int) -> np.ndarray:
        """重采样音频"""
        if original_rate == target_rate:
            return audio_data
        
        # 简单的重采样实现
        ratio = target_rate / original_rate
        new_length = int(len(audio_data) * ratio)
        
        # 使用线性插值
        indices = np.linspace(0, len(audio_data) - 1, new_length)
        resampled = np.interp(indices, np.arange(len(audio_data)), audio_data)
        
        return resampled.astype(audio_data.dtype)
    
    def normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """音频归一化"""
        if len(audio_data) == 0:
            return audio_data
        
        # 计算RMS
        rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
        
        if rms > 0:
            # 归一化到-1到1
            normalized = audio_data.astype(np.float32) / rms
            # 缩放到16位整数范围
            normalized = (normalized * 32767).astype(np.int16)
            return normalized
        
        return audio_data
    
    def save_audio(self, audio_data: np.ndarray, output_path: str, 
                  sample_rate: int, format: str = "wav") -> bool:
        """保存音频文件"""
        try:
            # 这里可以实现保存音频的逻辑
            # 暂时返回True
            return True
        except Exception as e:
            self.logger.error(f"保存音频时发生错误: {str(e)}")
            return False
    
    def close(self):
        """释放资源"""
        # FFmpeg处理器不需要特殊清理
        pass 
