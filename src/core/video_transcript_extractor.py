"""
视频转录提取器主类
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import logging

from .types import (
    ProcessingConfig, TranscriptionResult, TimeSegment, TranscriptionSegment
)
from .video_processor import VideoProcessor
from .audio_processor import AudioProcessor
from .vad_detector import VADDetector
from .ocr_processor import OCRProcessor
from .asr_processor import ASRProcessor
from .fusion_processor import FusionProcessor


class VideoTranscriptExtractor:
    """视频转录提取器主类"""
    
    def __init__(self, 
                 video_processor: Optional[VideoProcessor] = None,
                 audio_processor: Optional[AudioProcessor] = None,
                 vad_detector: Optional[VADDetector] = None,
                 ocr_processor: Optional[OCRProcessor] = None,
                 asr_processor: Optional[ASRProcessor] = None,
                 fusion_processor: Optional[FusionProcessor] = None,
                 config: Optional[ProcessingConfig] = None):
        """
        初始化视频转录提取器
        
        Args:
            video_processor: 视频处理器
            audio_processor: 音频处理器
            vad_detector: VAD检测器
            ocr_processor: OCR处理器
            asr_processor: ASR处理器
            fusion_processor: 融合处理器
            config: 处理配置
        """
        self.config = config or ProcessingConfig()
        self.video_processor = video_processor
        self.audio_processor = audio_processor
        self.vad_detector = vad_detector
        self.ocr_processor = ocr_processor
        self.asr_processor = asr_processor
        self.fusion_processor = fusion_processor
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 处理状态
        self._is_processing = False
        self._current_progress = 0.0
        
    def extract(self, video_path: str, 
               output_path: Optional[str] = None,
               save_intermediate: bool = False) -> TranscriptionResult:
        """
        从视频中提取转录
        
        Args:
            video_path: 视频文件路径
            output_path: 输出文件路径（可选）
            save_intermediate: 是否保存中间结果
            
        Returns:
            TranscriptionResult: 转录结果
        """
        self._is_processing = True
        self._current_progress = 0.0
        
        try:
            self.logger.info(f"开始处理视频: {video_path}")
            
            # 1. 加载视频
            self.logger.info("步骤 1/6: 加载视频")
            if not self._load_video(video_path):
                raise RuntimeError("视频加载失败")
            self._current_progress = 0.1
            
            # 2. 提取音频
            self.logger.info("步骤 2/6: 提取音频")
            audio_path = self._extract_audio(video_path)
            if not audio_path:
                raise RuntimeError("音频提取失败")
            self._current_progress = 0.2
            
            # 3. VAD检测语音区间
            self.logger.info("步骤 3/6: VAD检测语音区间")
            voice_segments = self._detect_voice_segments(audio_path)
            if not voice_segments:
                self.logger.warning("未检测到语音区间")
                return self._create_empty_result()
            self._current_progress = 0.4
            
            # 4. OCR识别硬字幕
            self.logger.info("步骤 4/6: OCR识别硬字幕")
            ocr_results = self._extract_ocr_results(voice_segments)
            self._current_progress = 0.6
            
            # 5. ASR语音识别
            self.logger.info("步骤 5/6: ASR语音识别")
            asr_results = self._extract_asr_results(voice_segments, audio_path)
            self._current_progress = 0.8
            
            # 6. 融合结果
            self.logger.info("步骤 6/6: 融合结果")
            final_segments = self._fuse_results(ocr_results, asr_results)
            self._current_progress = 0.9
            
            # 创建最终结果
            result = self._create_final_result(final_segments, video_path)
            self._current_progress = 1.0
            
            # 保存结果
            if output_path:
                self._save_result(result, output_path)
            
            self.logger.info("视频处理完成")
            return result
            
        except Exception as e:
            self.logger.error(f"处理过程中发生错误: {str(e)}")
            raise
        finally:
            self._is_processing = False
    
    def _load_video(self, video_path: str) -> bool:
        """加载视频"""
        if not self.video_processor:
            raise RuntimeError("未设置视频处理器")
        
        return self.video_processor.load_video(video_path)
    
    def _extract_audio(self, video_path: str) -> Optional[str]:
        """提取音频"""
        if not self.audio_processor:
            raise RuntimeError("未设置音频处理器")
        
        return self.audio_processor.extract_audio_from_video(video_path)
    
    def _detect_voice_segments(self, audio_path: str) -> List[TimeSegment]:
        """检测语音区间"""
        if not self.vad_detector:
            raise RuntimeError("未设置VAD检测器")
        
        return self.vad_detector.detect_voice_segments_from_file(audio_path)
    
    def _extract_ocr_results(self, voice_segments: List[TimeSegment]) -> List:
        """提取OCR结果"""
        if not self.ocr_processor:
            self.logger.warning("未设置OCR处理器，跳过OCR处理")
            return []
        
        if not self.video_processor:
            raise RuntimeError("未设置视频处理器")
        
        ocr_results = []
        for segment in voice_segments:
            # 提取该时间区间的视频帧
            frames = self.video_processor.extract_frames([(segment.start, segment.end)])
            if frames and frames[0]:
                # 进行OCR识别
                segment_results = self.ocr_processor.process_frames(frames[0])
                ocr_results.extend(segment_results)
        
        return ocr_results
    
    def _extract_asr_results(self, voice_segments: List[TimeSegment], 
                           audio_path: str) -> List:
        """提取ASR结果"""
        if not self.asr_processor:
            raise RuntimeError("未设置ASR处理器")
        
        if not self.audio_processor:
            raise RuntimeError("未设置音频处理器")
        
        # 提取语音区间的音频片段
        time_segments = [(seg.start, seg.end) for seg in voice_segments]
        audio_segments = self.audio_processor.extract_audio_segments(time_segments)
        
        # 进行ASR识别
        return self.asr_processor.process_audio_segments(audio_segments)
    
    def _fuse_results(self, ocr_results: List, asr_results: List) -> List:
        """融合结果"""
        if not self.fusion_processor:
            raise RuntimeError("未设置融合处理器")
        
        return self.fusion_processor.process_fusion(ocr_results, asr_results)
    
    def _create_empty_result(self) -> TranscriptionResult:
        """创建空结果"""
        from .types import TranscriptionSegment
        
        return TranscriptionResult(
            segments=[],
            full_text="",
            duration=0.0,
            metadata={"status": "no_voice_detected"}
        )
    
    def _create_final_result(self, segments: List, video_path: str) -> TranscriptionResult:
        """创建最终结果"""
        from .types import TranscriptionSegment
        
        # 获取视频信息
        duration = 0.0
        if self.video_processor:
            video_info = self.video_processor.get_video_info()
            duration = video_info.get('duration', 0.0)
        
        # 构建完整文本
        full_text = " ".join([seg.text for seg in segments])
        
        # 构建元数据
        metadata = {
            "video_path": video_path,
            "duration": duration,
            "segment_count": len(segments),
            "ocr_count": len([s for s in segments if s.source == 'ocr']),
            "asr_count": len([s for s in segments if s.source == 'asr']),
            "fusion_count": len([s for s in segments if s.source == 'fusion'])
        }
        
        return TranscriptionResult(
            segments=segments,
            full_text=full_text,
            duration=duration,
            metadata=metadata
        )
    
    def _save_result(self, result: TranscriptionResult, output_path: str):
        """保存结果"""
        # 这里可以实现保存为不同格式的逻辑
        # 如JSON、SRT、VTT等
        pass
    
    @property
    def is_processing(self) -> bool:
        """是否正在处理"""
        return self._is_processing
    
    @property
    def progress(self) -> float:
        """当前进度 (0-1)"""
        return self._current_progress
    
    def set_processors(self, 
                      video_processor: Optional[VideoProcessor] = None,
                      audio_processor: Optional[AudioProcessor] = None,
                      vad_detector: Optional[VADDetector] = None,
                      ocr_processor: Optional[OCRProcessor] = None,
                      asr_processor: Optional[ASRProcessor] = None,
                      fusion_processor: Optional[FusionProcessor] = None):
        """设置处理器"""
        if video_processor:
            self.video_processor = video_processor
        if audio_processor:
            self.audio_processor = audio_processor
        if vad_detector:
            self.vad_detector = vad_detector
        if ocr_processor:
            self.ocr_processor = ocr_processor
        if asr_processor:
            self.asr_processor = asr_processor
        if fusion_processor:
            self.fusion_processor = fusion_processor
    
    def get_config(self) -> ProcessingConfig:
        """获取配置"""
        return self.config
    
    def set_config(self, config: ProcessingConfig):
        """设置配置"""
        self.config = config 