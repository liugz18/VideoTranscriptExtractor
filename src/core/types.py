"""
核心数据类型定义
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import numpy as np


@dataclass
class TimeSegment:
    """时间区间定义"""
    start: float  # 开始时间（秒）
    end: float    # 结束时间（秒）
    
    @property
    def duration(self) -> float:
        """区间持续时间"""
        return self.end - self.start
    
    def overlaps(self, other: 'TimeSegment') -> bool:
        """检查是否与其他区间重叠"""
        return not (self.end <= other.start or other.end <= self.start)


@dataclass
class VideoFrame:
    """视频帧数据"""
    timestamp: float  # 时间戳（秒）
    frame_data: np.ndarray  # 帧图像数据
    frame_number: int  # 帧序号


@dataclass
class AudioSegment:
    """音频片段数据"""
    start_time: float  # 开始时间（秒）
    end_time: float    # 结束时间（秒）
    audio_data: np.ndarray  # 音频数据
    sample_rate: int  # 采样率


@dataclass
class OCRResult:
    """OCR识别结果"""
    text: str  # 识别的文本
    confidence: float  # 置信度 (0-1)
    bbox: Optional[List[float]] = None  # 边界框 [x1, y1, x2, y2]
    timestamp: Optional[float] = None  # 时间戳


@dataclass
class ASRResult:
    """ASR识别结果"""
    text: str  # 识别的文本
    confidence: float  # 置信度 (0-1)
    start_time: float  # 开始时间
    end_time: float    # 结束时间
    words: Optional[List[Dict[str, Any]]] = None  # 词级别时间戳


@dataclass
class TranscriptionSegment:
    """转录片段"""
    start_time: float  # 开始时间
    end_time: float    # 结束时间
    text: str  # 转录文本
    source: str  # 数据来源 ('ocr', 'asr', 'fusion')
    confidence: float  # 置信度


@dataclass
class TranscriptionResult:
    """完整的转录结果"""
    segments: List[TranscriptionSegment]  # 转录片段列表
    full_text: str  # 完整文本
    duration: float  # 视频总时长
    metadata: Dict[str, Any]  # 元数据
    
    @property
    def transcript(self) -> str:
        """获取完整转录文本"""
        return self.full_text


@dataclass
class ProcessingConfig:
    """处理配置"""
    # VAD配置
    vad_threshold: float = 0.5
    vad_min_duration: float = 0.5
    vad_max_duration: float = 10.0
    
    # OCR配置
    ocr_confidence_threshold: float = 0.7
    ocr_languages: List[str] = None
    
    # ASR配置
    asr_model: str = "whisper"
    asr_language: str = "auto"
    
    # 融合配置
    fusion_weight_ocr: float = 0.6
    fusion_weight_asr: float = 0.4
    
    def __post_init__(self):
        if self.ocr_languages is None:
            self.ocr_languages = ["en", "zh"] 