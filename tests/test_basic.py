"""
基本测试
"""

import pytest
from src.core import ProcessingConfig, TimeSegment, VideoFrame, AudioSegment
from src.core import OCRResult, ASRResult, TranscriptionSegment, TranscriptionResult


def test_processing_config():
    """测试处理配置"""
    config = ProcessingConfig()
    
    assert config.vad_threshold == 0.5
    assert config.vad_min_duration == 0.5
    assert config.vad_max_duration == 10.0
    assert config.ocr_confidence_threshold == 0.7
    assert config.asr_model == "whisper"
    assert config.asr_language == "auto"
    assert config.fusion_weight_ocr == 0.6
    assert config.fusion_weight_asr == 0.4
    assert config.ocr_languages == ["en", "zh"]


def test_time_segment():
    """测试时间区间"""
    segment = TimeSegment(start=1.0, end=5.0)
    
    assert segment.start == 1.0
    assert segment.end == 5.0
    assert segment.duration == 4.0
    
    # 测试重叠检测
    other = TimeSegment(start=3.0, end=7.0)
    assert segment.overlaps(other) == True
    
    other2 = TimeSegment(start=6.0, end=8.0)
    assert segment.overlaps(other2) == False


def test_video_frame():
    """测试视频帧"""
    import numpy as np
    
    frame_data = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = VideoFrame(
        timestamp=1.5,
        frame_data=frame_data,
        frame_number=100
    )
    
    assert frame.timestamp == 1.5
    assert frame.frame_number == 100
    assert frame.frame_data.shape == (480, 640, 3)


def test_audio_segment():
    """测试音频片段"""
    import numpy as np
    
    audio_data = np.zeros(16000, dtype=np.int16)  # 1秒音频
    segment = AudioSegment(
        start_time=0.0,
        end_time=1.0,
        audio_data=audio_data,
        sample_rate=16000
    )
    
    assert segment.start_time == 0.0
    assert segment.end_time == 1.0
    assert len(segment.audio_data) == 16000
    assert segment.sample_rate == 16000


def test_ocr_result():
    """测试OCR结果"""
    result = OCRResult(
        text="Hello World",
        confidence=0.8,
        bbox=[10, 20, 100, 50],
        timestamp=2.5
    )
    
    assert result.text == "Hello World"
    assert result.confidence == 0.8
    assert result.bbox == [10, 20, 100, 50]
    assert result.timestamp == 2.5


def test_asr_result():
    """测试ASR结果"""
    result = ASRResult(
        text="Hello World",
        confidence=0.9,
        start_time=1.0,
        end_time=3.0,
        words=[{"word": "Hello", "start": 1.0, "end": 2.0}]
    )
    
    assert result.text == "Hello World"
    assert result.confidence == 0.9
    assert result.start_time == 1.0
    assert result.end_time == 3.0
    assert len(result.words) == 1


def test_transcription_segment():
    """测试转录片段"""
    segment = TranscriptionSegment(
        start_time=1.0,
        end_time=3.0,
        text="Hello World",
        source="fusion",
        confidence=0.85
    )
    
    assert segment.start_time == 1.0
    assert segment.end_time == 3.0
    assert segment.text == "Hello World"
    assert segment.source == "fusion"
    assert segment.confidence == 0.85


def test_transcription_result():
    """测试转录结果"""
    segments = [
        TranscriptionSegment(
            start_time=0.0,
            end_time=2.0,
            text="Hello",
            source="ocr",
            confidence=0.8
        ),
        TranscriptionSegment(
            start_time=2.0,
            end_time=4.0,
            text="World",
            source="asr",
            confidence=0.9
        )
    ]
    
    result = TranscriptionResult(
        segments=segments,
        full_text="Hello World",
        duration=4.0,
        metadata={"test": "value"}
    )
    
    assert len(result.segments) == 2
    assert result.full_text == "Hello World"
    assert result.transcript == "Hello World"
    assert result.duration == 4.0
    assert result.metadata["test"] == "value"


if __name__ == "__main__":
    pytest.main([__file__]) 