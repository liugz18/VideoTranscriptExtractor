"""
具体处理器实现模块
"""

from .video.opencv_processor import OpenCVVideoProcessor
from .audio.ffmpeg_processor import FFmpegAudioProcessor
from .vad.webrtc_vad_processor import WebRTCVADProcessor
from .ocr.easyocr_processor import EasyOCRProcessor
from .asr.whisper_processor import WhisperASRProcessor
from .fusion.simple_fusion_processor import SimpleFusionProcessor

__all__ = [
    "OpenCVVideoProcessor",
    "FFmpegAudioProcessor", 
    "WebRTCVADProcessor",
    "EasyOCRProcessor",
    "WhisperASRProcessor",
    "SimpleFusionProcessor"
] 