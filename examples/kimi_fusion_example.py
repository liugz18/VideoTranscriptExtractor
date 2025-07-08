from src.core.types import TranscriptionResult
from src.processors.fusion.kimi_fusion_processor import KimiFusionProcessor
from src.processors.audio.ffmpeg_processor import FFmpegAudioProcessor
from src.processors.vad.silero_vad_processor import SileroVADProcessor
import os

# 路径根据实际情况调整
ocr_json = "output/[OCR]20250626上海话视频480568590_nb2-1-16_transcription.json"
video_path = "sample_data/20250626上海话视频480568590_nb2-1-16.mp4"
output_json = "output/[KIMI]20250626上海话视频480568590_nb2-1-16_transcription.json"


def main():
    # 1. 读取OCR结果
    ocr_result = TranscriptionResult.load_from_file(ocr_json)
    ocr_segments = ocr_result.segments

    # 2. 提取视频音频
    audio_processor = FFmpegAudioProcessor()
    audio_path = audio_processor.extract_audio_from_video(video_path)
    assert audio_path is not None, "音频提取失败"

    # 3. VAD分段
    vad_processor = SileroVADProcessor()
    vad_segments = vad_processor.detect_voice_segments_from_file(audio_path)
    print(f"检测到 {len(vad_segments)} 个语音区间")
    time_tuples = [(segment.start, segment.end) for segment in vad_segments]

    # 4. 切割音频
    audio_segments = audio_processor.extract_audio_segments(time_tuples, audio_path)
    audio_file_paths = [seg.path for seg in audio_segments]


    # 5. 对齐OCR和音频片段数量
    min_len = min(len(ocr_segments), len(audio_file_paths))
    ocr_segments = ocr_segments[:min_len]
    audio_file_paths = audio_file_paths[:min_len]

    # 6. 融合
    kimi_fusion = KimiFusionProcessor()
    fused_result = kimi_fusion.fuse_results(ocr_segments, audio_file_paths)

    # 7. 保存结果
    fused_result.save_to_file(output_json, format="json")
    print(f"Kimi融合结果已保存到: {output_json}")

if __name__ == "__main__":
    main() 