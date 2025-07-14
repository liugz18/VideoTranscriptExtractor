from typing import Any, Optional
from src.core.fusion_processor import FusionProcessor
from src.core.types import TranscriptionSegment, TranscriptionResult
import requests
import json

class LLMAPIFusionProcessor(FusionProcessor):
    """基于LLM的字幕融合处理器"""
    def __init__(self, llm_client: Any, prompt: Optional[str] = None):
        """
        llm_client: 用于生成融合字幕的LLM客户端，需有generate(prompt)方法
        prompt: 默认融合用的提示词，可自定义
        """
        self.llm_client = llm_client
        self.prompt = prompt if prompt is not None else \
        (
            "现在有两个来源的字幕片段需要融合为最准确的字幕：\n"
            "OCR结果包含正确的字幕，但会把图片中的不同识别结果都用|拼接起来，可能夹杂其他视频字符且可能重复。\n"
            "ASR结果字数比较对但可能识别成同音字较多。\n"
            "请注意，根据ASR文本的长度和发音，在OCR文本中摘取相应发音的片段即可。\n"
            "如果OCR缺少个别字，从ASR文本中摘取补足。\n"
            "要过滤掉OCR文本中没有在ASR文本里找到对应发音的部分。\n"
            "仅输出融合后的字幕内容，不要解释。\n"
            "样例："
            "【OCR识别结果】\n新上海人|大哥|新上海人|你放心讲|新上海人|我帮他翻译\n"
            "【ASR识别结果】\n阿阔您放心搞我帮他翻翻\n"
            "【融合结果】\n大哥你放心讲我帮他翻译\n\n"
            "需要你做的："
            "【OCR识别结果】\n{ocr}\n"
            "【ASR识别结果】\n{asr}\n"
            "【融合结果（你的输出）】\n"
        )
        

    def fuse_segment(self, seg1: TranscriptionSegment, seg2: TranscriptionSegment, api_token: str) -> TranscriptionSegment:
        """
        使用外部API进行字幕融合，返回一个新的TranscriptionSegment。
        通过POST请求 https://api.siliconflow.cn/v1/chat/completions。
        """
        url = "https://api.siliconflow.cn/v1/chat/completions"
        prompt = self.prompt.format(ocr=seg1.text, asr=seg2.text)
        payload = {
            "model": "deepseek-ai/DeepSeek-V3",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        try:
            response = requests.request("POST", url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            result = json.loads(response.text)
            if "choices" in result and result["choices"] and "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
                fused_text = result["choices"][0]["message"]["content"]
                print(fused_text)
            else:
                fused_text = "[融合失败: 未返回内容]"
        except Exception as e:
            fused_text = f"[融合失败: {e}]"
        return TranscriptionSegment(
            start_time=min(seg1.start_time, seg2.start_time),
            end_time=max(seg1.end_time, seg2.end_time),
            text=fused_text,
            source="llm_fusion_api",
            confidence=max(seg1.confidence, seg2.confidence)
        ) 

    def fuse_results(self, result1: TranscriptionResult, result2: TranscriptionResult, api_token: str) -> TranscriptionResult:
        """
        对两个TranscriptionResult的segments zip后融合，返回新的TranscriptionResult。
        """
        fused_segments = []
        for seg1, seg2 in zip(result1.segments, result2.segments):
            fused_segments.append(self.fuse_segment(seg1, seg2, api_token))
        duration = max(result1.duration, result2.duration)
        metadata = {"fusion": "llm", **result1.metadata, **result2.metadata}
        return TranscriptionResult(
            segments=fused_segments,
            duration=duration,
            metadata=metadata
        ) 