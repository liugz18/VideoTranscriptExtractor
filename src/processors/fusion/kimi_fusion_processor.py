import os
import sys
kimi_root_path = "/mnt/sda/20250403来自HDD的备份/YuYinDuoMoTai/Kimi-Audio"
sys.path.append(kimi_root_path)
from typing import List, Dict, Any, Optional
from src.core.types import TranscriptionSegment, TranscriptionResult
from kimia_infer.api.kimia import KimiAudio


class KimiFusionProcessor:
    """基于Kimi-Audio大模型的多模态融合处理器"""
    def __init__(self, model_path=f"{kimi_root_path}/Kimi-Audio-7B-Instruct", sampling_params: Optional[Dict[str, Any]] = None):
        self.model = KimiAudio(model_path=model_path, load_detokenizer=False)
        self.sampling_params = sampling_params or {
            "audio_temperature": 0.8,
            "audio_top_k": 10,
            "text_temperature": 0.0,
            "text_top_k": 5,
            "audio_repetition_penalty": 1.0,
            "audio_repetition_window_size": 64,
            "text_repetition_penalty": 1.0,
            "text_repetition_window_size": 16,
        }

    def fuse_result(self, ocr_seg: TranscriptionSegment, audio_path: str) -> TranscriptionSegment:
        """
        融合单个OCR片段和对应音频，返回融合后的TranscriptionSegment
        """
        messages = [
            {"role": "user", "message_type": "text", "content": f"请根据以下OCR结果再听一遍音频，在OCR文本中摘取相应发音的片段，转写出最准确的文本：{ocr_seg.text}"},
            {"role": "user", "message_type": "audio", "content": audio_path},
        ]
        try:
            wav, text = self.model.generate(messages, **self.sampling_params, output_type="text")
        except Exception as e:
            text = f"[Kimi融合失败: {e}]"
        fused_seg = TranscriptionSegment(
            start_time=ocr_seg.start_time,
            end_time=ocr_seg.end_time,
            text=text,
            source="kimi_fusion",
            confidence=ocr_seg.confidence  # 可根据需要自定义
        )
        print(fused_seg)
        return fused_seg

    def fuse_results(self, ocr_segments: List[TranscriptionSegment], audio_file_paths: List[str]) -> TranscriptionResult:
        """
        输入：
            ocr_segments: OCR识别的TranscriptionSegment列表
            audio_file_paths: 与ocr_segments一一对应的音频文件路径
        输出：
            融合后的TranscriptionResult
        """
        assert len(ocr_segments) == len(audio_file_paths), "ocr_segments和audio_file_paths长度需一致"
        fused_segments = []
        for ocr_seg, audio_path in zip(ocr_segments, audio_file_paths):
            fused_seg = self.fuse_result(ocr_seg, audio_path)
            fused_segments.append(fused_seg)
        duration = max(seg.end_time for seg in ocr_segments) if ocr_segments else 0.0
        metadata = {"fusion": "kimi-audio"}
        return TranscriptionResult(
            segments=fused_segments,
            duration=duration,
            metadata=metadata
        ) 