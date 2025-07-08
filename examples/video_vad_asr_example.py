"""
VAD+ASR对一个视频文件进行硬字幕提取：

1. 加载视频文件；
2. 提取视频中的音频轨道；
3. 使用语音活动检测（VAD）对音频进行分段，检测出语音区间；
4. 针对每个语音区间，进行切割成单个音频；
5. 对提取的每个音频进行ASR的识别。

"""
from src.processors.video.opencv_processor import OpenCVVideoProcessor
from src.processors.audio.ffmpeg_processor import FFmpegAudioProcessor
from src.processors.vad.silero_vad_processor import SileroVADProcessor

from src.processors.asr.paraformer_processor import ParaformerASRProcessor
from src.core.types import ProcessingConfig
from src.core.types import ProcessingConfig, TranscriptionResult, TranscriptionSegment


def main():
#    video_path = 'sample_data/20250626上海话视频480568590_nb2-1-16.mp4'
    video_path = 'sample_data/20250626上海话视频480568590_nb2-1-16.mp4'
    config = ProcessingConfig(ocr_languages=["ch"])

    video_processor = OpenCVVideoProcessor(config)
    audio_processor = FFmpegAudioProcessor(config)
    vad_processor = SileroVADProcessor(config)

    asr_processor = ParaformerASRProcessor(config,
            model_file="/mnt/sda/ASR/model/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch")#"/mnt/sda/ASR/zhanghui/FunASR/inference_model/secondmodel/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-jingzhou")
    # 1. 加载视频
    print("加载视频...")
    assert video_processor.load_video(video_path), "视频加载失败"

    # 2. 提取音频
    print("提取音频...")
    audio_path = audio_processor.extract_audio_from_video(video_path)
    assert audio_path is not None, "音频提取失败"

    # 3. VAD检测区间
    print("VAD检测...")
    segments = vad_processor.detect_voice_segments_from_file(audio_path)
    print(f"检测到 {len(segments)} 个语音区间")
    time_tuples = [(segment.start, segment.end) for segment in segments]
    # 4.切割音频
    print("按照vad切割音频...")
    audio_segments = audio_processor.extract_audio_segments(time_tuples,audio_path)
    all_segments = []
  # 5. ASR识别中
    for i, seg in enumerate(audio_segments):
        print(f"i:{i},seg:{seg}")
        frame = seg
        if frame is not None:
            asr_results = asr_processor.transcribe_audio(frame.audio_data,sample_rate=16000)
            if asr_results:
            # 合并文本
                combined_text = "|".join([r.text for r in asr_results])
                avg_confidence = sum(r.confidence for r in asr_results) / len(asr_results)

            # 构建 TranscriptionSegment
                segment = TranscriptionSegment(
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    text=combined_text,
                    source="asr",
                    confidence=avg_confidence
                )
                all_segments.append(segment)

                print(f"区间{i + 1}: {seg.start_time:.2f}-{seg.end_time:.2f}s, ASR: {combined_text}")
            else:
                print(f"区间{i + 1}: {seg.start_time:.2f}-{seg.end_time:.2f}s, 未提取到ASR结果")
        if all_segments:
            video_duration = video_processor.get_video_info()['duration']
            transcription_result = TranscriptionResult(
                segments=all_segments,
                duration=video_duration,  # 你需要提前算好
                metadata={
                    "source": "asr",
                    "audio_path": audio_path,
                    "total_segments": len(all_segments)
                }
            )
            import os
            audio_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = f"output/[ASR]{audio_name}_transcription.json"
            transcription_result.save_to_file(output_path, format="json")
            print(f"ASR 转录结果已保存到: {output_path}")
        else:
            print("未检测到任何 ASR 结果")
    video_processor.close()


if __name__ == "__main__":
    main() 
