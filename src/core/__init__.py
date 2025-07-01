"""
核心接口定义模块
"""

from .types import (
    TimeSegment, VideoFrame, AudioSegment, OCRResult, ASRResult,
    TranscriptionSegment, TranscriptionResult, ProcessingConfig
)
from .video_processor import VideoProcessor
from .audio_processor import AudioProcessor
from .vad_detector import VADDetector
from .ocr_processor import OCRProcessor
from .asr_processor import ASRProcessor
from .fusion_processor import FusionProcessor
from .video_transcript_extractor import VideoTranscriptExtractor

__all__ = [
    "TimeSegment", "VideoFrame", "AudioSegment", "OCRResult", "ASRResult",
    "TranscriptionSegment", "TranscriptionResult", "ProcessingConfig",
    "VideoProcessor",
    "AudioProcessor", 
    "VADDetector",
    "OCRProcessor",
    "ASRProcessor",
    "FusionProcessor",
    "VideoTranscriptExtractor"
] 