"""
SileroVADProcessor 测试
"""
import pytest
from src.processors.vad.silero_vad_processor import SileroVADProcessor
from src.core.types import TimeSegment


def test_silero_vad_on_real_file():
    # 假设sample_data/1417796_input.wav_10.wav存在
    audio_path = 'sample_data/1417796_input.wav_10.wav'
    processor = SileroVADProcessor()
    segments = processor.detect_voice_segments_from_file(audio_path)
    assert isinstance(segments, list)
    assert all(isinstance(seg, TimeSegment) for seg in segments)
    # 结果应与官方样例类似
    assert any(abs(seg.start - 0.3) < 0.2 and abs(seg.end - 3.8) < 0.2 for seg in segments)
    print([{'start': seg.start, 'end': seg.end} for seg in segments]) 