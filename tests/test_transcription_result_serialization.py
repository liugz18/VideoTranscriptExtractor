import os
import pytest
from src.core import TranscriptionResult

def test_transcription_result_serialization():
    input_path = 'output/20250626上海话视频480568590_nb2-1-16_transcription.json'
    temp_path = 'output/test.json'

    # 反序列化
    result1 = TranscriptionResult.load_from_file(input_path)
    # 序列化
    result1.save_to_file(temp_path, format='json')
    # 再次反序列化
    result2 = TranscriptionResult.load_from_file(temp_path)

    # 验证全等
    assert result1.segments == result2.segments
    assert result1.duration == result2.duration
    assert result1.metadata == result2.metadata

    # 清理临时文件
    os.remove(temp_path) 