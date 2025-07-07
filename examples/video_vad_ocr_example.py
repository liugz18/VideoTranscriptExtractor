"""
VAD+OCR对一个视频文件进行硬字幕提取：

1. 加载视频文件；
2. 提取视频中的音频轨道；
3. 使用语音活动检测（VAD）对音频进行分段，检测出语音区间；
4. 每个区间采样多帧做OCR，检测字幕变化；
5. 将所有OCR结果合并，输出字幕文本。
"""
from src.processors.video.opencv_processor import OpenCVVideoProcessor
from src.processors.audio.ffmpeg_processor import FFmpegAudioProcessor
from src.processors.vad.silero_vad_processor import SileroVADProcessor
from src.processors.ocr.paddleocr_processor import PaddleOCRProcessor
from src.core.types import ProcessingConfig, TranscriptionResult, TranscriptionSegment


def main():
   # video_path ='sample_data/阴阳怪气的周姐10 #搞笑 #高考查分 #真实 #反转 #武汉话.mp4'
    video_path = 'sample_data/20250626上海话视频480568590_nb2-1-16.mp4'#
    config = ProcessingConfig(ocr_languages=["ch"])
    
    video_processor = OpenCVVideoProcessor(config)
    audio_processor = FFmpegAudioProcessor(config)
    vad_processor = SileroVADProcessor(config)
    ocr_processor = PaddleOCRProcessor(config)

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
    
    # 4. 每个区间采样多帧做OCR，检测字幕变化
    sample_interval = 20  # 每n帧采样一次
    all_segments = []
    
    for i, seg in enumerate(segments):
        frames_list = video_processor.extract_frames([(seg.start, seg.end)])
        frames = frames_list[0] if frames_list else []
        if frames:
            ocr_results = ocr_processor.recognize_text_from_frames(frames, sample_interval=sample_interval)
            texts = [r.text for r in ocr_results]
            print(f"区间{i+1}: {seg.start:.2f}-{seg.end:.2f}s, OCR: {texts}")
            
            # 将OCR结果转换为TranscriptionSegment
            if ocr_results:
                # 合并同一区间的所有文本
                combined_text = "|".join(texts)
                avg_confidence = sum(r.confidence for r in ocr_results) / len(ocr_results)
                
                segment = TranscriptionSegment(
                    start_time=seg.start,
                    end_time=seg.end,
                    text=combined_text,
                    source="ocr",
                    confidence=avg_confidence
                )
                all_segments.append(segment)
        else:
            print(f"区间{i+1}: {seg.start:.2f}-{seg.end:.2f}s, 未提取到帧")
    
    # 5. 创建TranscriptionResult对象并保存
    if all_segments:
        # 获取视频总时长
        video_duration = video_processor.get_video_info()['duration']
        
        
        # 创建转录结果对象
        transcription_result = TranscriptionResult(
            segments=all_segments,
            duration=video_duration,
            metadata={
                "source": "ocr",
                "video_path": video_path,
                "sample_interval": sample_interval,
                "total_segments": len(all_segments)
            }
        )
        
        # 保存到JSON文件
        import os
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"output/[OCR]{video_name}_transcription.json"
        transcription_result.save_to_file(output_path, format="json")
        print(f"转录结果已保存到: {output_path}")
    else:
        print("未检测到任何OCR结果")

    video_processor.close()


if __name__ == "__main__":
    main() 
