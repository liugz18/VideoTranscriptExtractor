from src.core.types import TranscriptionResult
from src.processors.fusion.llm_fusion_processor import LLMFusionProcessor
import os

# 路径根据实际情况调整
ocr_json = "output/[OCR]20250626上海话视频480568590_nb2-1-16_transcription.json"
asr_json = "output/[ASR]20250626上海话视频480568590_nb2-1-16_transcription.json"
output_json = "output/[FUSION]20250626上海话视频480568590_nb2-1-16_transcription.json"

def main():
    # 加载OCR和ASR转录结果
    ocr_result = TranscriptionResult.load_from_file(ocr_json)
    asr_result = TranscriptionResult.load_from_file(asr_json)

    # 取最短长度，避免越界
    min_len = min(len(ocr_result.segments), len(asr_result.segments))
    ocr_segments = ocr_result.segments[:min_len]
    asr_segments = asr_result.segments[:min_len]

    # 初始化融合处理器（llm_client可为None，因fuse_segment内部未用到）
    fusion_processor = LLMFusionProcessor(llm_client=None)

    # 融合所有segment
    fused_segments = []
    for i, (ocr_seg, asr_seg) in enumerate(zip(ocr_segments, asr_segments)):
        fused_seg = fusion_processor.fuse_segment(ocr_seg, asr_seg)
        fused_segments.append(fused_seg)
        print(f"第{i+1}段融合完成")
        print(fused_seg)

    # 构建新的TranscriptionResult
    fused_result = TranscriptionResult(
        segments=fused_segments,
        duration=max(ocr_result.duration, asr_result.duration),
        metadata={"fusion": "llm", **ocr_result.metadata, **asr_result.metadata}
    )

    # 保存为新的transcription.json
    fused_result.save_to_file(output_json, format="json")
    print(f"融合结果已保存到: {output_json}")

if __name__ == "__main__":
    main() 