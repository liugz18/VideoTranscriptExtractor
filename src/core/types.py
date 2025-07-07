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
    duration: float  # 视频总时长
    metadata: Dict[str, Any]  # 元数据
    
    @property
    def transcript(self) -> str:
        """获取完整转录文本"""
        return " ".join(seg.text for seg in self.segments)
    
    def save_to_file(self, file_path: str, format: str = "json") -> None:
        """
        将转录结果序列化并保存到文件
        
        Args:
            file_path: 文件路径
            format: 文件格式 ("json" 或 "txt")
        """
        import json
        import os
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if format.lower() == "json":
            # 保存为JSON格式，包含完整信息
            data = {
                "segments": [
                    {
                        "start_time": seg.start_time,
                        "end_time": seg.end_time,
                        "text": seg.text,
                        "source": seg.source,
                        "confidence": seg.confidence
                    }
                    for seg in self.segments
                ],
                "duration": self.duration,
                "metadata": self.metadata
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        elif format.lower() == "txt":
            # 保存为纯文本格式
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# 转录结果\n")
                f.write(f"# 总时长: {self.duration:.2f}秒\n")
                f.write(f"# 片段数: {len(self.segments)}\n\n")
                
                for i, seg in enumerate(self.segments, 1):
                    f.write(f"[{seg.start_time:.2f}s - {seg.end_time:.2f}s] ({seg.source})\n")
                    f.write(f"{seg.text}\n\n")
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'TranscriptionResult':
        """
        从文件反序列化转录结果
        
        Args:
            file_path: 文件路径
            
        Returns:
            TranscriptionResult: 转录结果对象
        """
        import json
        import os
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        if file_path.lower().endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            segments = [
                TranscriptionSegment(
                    start_time=seg["start_time"],
                    end_time=seg["end_time"],
                    text=seg["text"],
                    source=seg["source"],
                    confidence=seg["confidence"]
                )
                for seg in data["segments"]
            ]
            
            return cls(
                segments=segments,
                duration=data["duration"],
                metadata=data["metadata"]
            )
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")


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