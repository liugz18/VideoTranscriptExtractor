"""
基于Silero VAD的VAD处理器实现
"""

import numpy as np
from typing import List, Optional
import logging

from ...core.vad_detector import VADDetector
from ...core.types import TimeSegment, AudioSegment, ProcessingConfig

class SileroVADProcessor(VADDetector):
    """基于Silero VAD的VAD处理器"""
    def __init__(self, config: Optional[ProcessingConfig] = None):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            from silero_vad import load_silero_vad
            self.model = load_silero_vad()
        except ImportError:
            self.logger.warning("silero-vad未安装，使用模拟VAD")
            self.model = None

    def detect_voice_segments(self, audio_data: np.ndarray, sample_rate: int) -> List[TimeSegment]:
        if self.model is None:
            return self._simulate_vad_detection(audio_data, sample_rate)
        try:
            from silero_vad import get_speech_timestamps
            # Silero VAD 需要float32, -1~1
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32) / np.max(np.abs(audio_data))
            speech_timestamps = get_speech_timestamps(
                audio_data, self.model, sampling_rate=sample_rate, return_seconds=True
            )
            segments = [TimeSegment(start=ts['start'], end=ts['end']) for ts in speech_timestamps]
            return segments
        except Exception as e:
            self.logger.error(f"Silero VAD检测时发生错误: {str(e)}")
            return self._simulate_vad_detection(audio_data, sample_rate)

    def detect_voice_segments_from_file(self, audio_path: str) -> List[TimeSegment]:
        if self.model is None:
            return []
        try:
            from silero_vad import read_audio, get_speech_timestamps
            wav = read_audio(audio_path)
            speech_timestamps = get_speech_timestamps(
                wav, self.model, return_seconds=True, min_silence_duration_ms=40
            )
            segments = [TimeSegment(start=ts['start'], end=ts['end']) for ts in speech_timestamps]
            return segments
        except Exception as e:
            self.logger.error(f"Silero VAD文件检测时发生错误: {str(e)}")
            return []

    def _simulate_vad_detection(self, audio_data: np.ndarray, sample_rate: int) -> List[TimeSegment]:
        # 简单模拟
        duration = len(audio_data) / sample_rate
        return [TimeSegment(start=0.0, end=duration)]

    def detect_voice_segments_from_segments(self, audio_segments: List[AudioSegment]) -> List[TimeSegment]:
        segments = []
        for segment in audio_segments:
            segment_results = self.detect_voice_segments(segment.audio_data, segment.sample_rate)
            for result in segment_results:
                adjusted_segment = TimeSegment(
                    start=segment.start_time + result.start,
                    end=segment.start_time + result.end
                )
                segments.append(adjusted_segment)
        return segments

    def filter_segments(self, segments: List[TimeSegment], min_duration: Optional[float] = None, max_duration: Optional[float] = None) -> List[TimeSegment]:
        if min_duration is None:
            min_duration = self.config.vad_min_duration
        if max_duration is None:
            max_duration = self.config.vad_max_duration
        return [s for s in segments if min_duration <= s.duration <= max_duration]

    def merge_adjacent_segments(self, segments: List[TimeSegment], gap_threshold: float = 0.5) -> List[TimeSegment]:
        if not segments:
            return []
        sorted_segments = sorted(segments, key=lambda x: x.start)
        merged = [sorted_segments[0]]
        for segment in sorted_segments[1:]:
            last = merged[-1]
            if segment.start - last.end <= gap_threshold:
                merged[-1] = TimeSegment(start=last.start, end=max(last.end, segment.end))
            else:
                merged.append(segment)
        return merged

    def get_voice_probability(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        # Silero VAD不直接输出概率，这里返回全1
        return np.ones_like(audio_data, dtype=np.float32) 