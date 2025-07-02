from typing import Any, Optional
from src.core.fusion_processor import FusionProcessor
from src.core.types import TranscriptionSegment, TranscriptionResult

class LLMFusionProcessor(FusionProcessor):
    """基于LLM的字幕融合处理器"""
    def __init__(self, llm_client: Any, prompt: Optional[str] = None):
        """
        llm_client: 用于生成融合字幕的LLM客户端，需有generate(prompt)方法
        prompt: 默认融合用的提示词，可自定义
        """
        self.llm_client = llm_client
        self.prompt = prompt if prompt is not None else \
        (
            "你是一个视频字幕融合专家。\n"
            "现在有两个来源的字幕片段需要融合为最准确的字幕：\n"
            "OCR结果通常包含大致正确的字幕，但可能夹杂其他视频字符且可能重复。\n"
            "ASR结果字数比较对但可能识别成同音字较多。\n"
            "请你综合两者内容，输出最准确、自然的字幕文本，仅输出融合后的字幕内容，不要解释。\n"
            "【OCR识别结果】\n{ocr}\n"
            "【ASR识别结果】\n{asr}\n"
            "【融合结果】\n"
        )
        

    def fuse_segment(self, seg1: TranscriptionSegment, seg2: TranscriptionSegment) -> TranscriptionSegment:
        """
        融合两个TranscriptionSegment，返回一个新的TranscriptionSegment。
        """
        # 构造融合Prompt
        prompt = self.prompt.format(ocr=seg1.text, asr=seg2.text)
        # 调用LLM接口（假设self.llm_client有一个generate方法）
        fused_text = self.llm_client.generate(prompt)
        return TranscriptionSegment(
            start_time=min(seg1.start_time, seg2.start_time),
            end_time=max(seg1.end_time, seg2.end_time),
            text=fused_text,
            source="llm_fusion",
            confidence=max(seg1.confidence, seg2.confidence)
        )

    def fuse_results(self, result1: TranscriptionResult, result2: TranscriptionResult) -> TranscriptionResult:
        """
        对两个TranscriptionResult的segments zip后融合，返回新的TranscriptionResult。
        """
        fused_segments = []
        for seg1, seg2 in zip(result1.segments, result2.segments):
            fused_segments.append(self.fuse_segment(seg1, seg2))
        full_text = " ".join([seg.text for seg in fused_segments])
        duration = max(result1.duration, result2.duration)
        metadata = {"fusion": "llm", **result1.metadata, **result2.metadata}
        return TranscriptionResult(
            segments=fused_segments,
            full_text=full_text,
            duration=duration,
            metadata=metadata
        ) 