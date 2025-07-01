"""
VAD+OCR对一个视频文件进行硬字幕提取：

1. 加载视频文件；
2. 提取视频中的音频轨道；
3. 使用语音活动检测（VAD）对音频进行分段，检测出语音区间；
4. 针对每个语音区间，提取该区间中点对应的视频帧；
5. 对提取的帧进行OCR文字识别，获取视频画面中的文字内容。

"""
from src.processors.video.opencv_processor import OpenCVVideoProcessor
from src.processors.audio.ffmpeg_processor import FFmpegAudioProcessor
from src.processors.vad.silero_vad_processor import SileroVADProcessor
from src.processors.ocr.paddleocr_processor import PaddleOCRProcessor
from src.core.types import ProcessingConfig


def main():
    video_path = 'sample_data/20250626上海话视频480568590_nb2-1-16.mp4'
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
    
    # 4. 每个区间的中点帧做OCR
    for i, seg in enumerate(segments):
        mid_time = (seg.start + seg.end) / 2
        frame = video_processor.extract_frame_at_time(mid_time)
        if frame is not None:
            ocr_results = ocr_processor.recognize_text(frame.frame_data)
            texts = [r.text for r in ocr_results]
            print(f"区间{i+1}: {seg.start:.2f}-{seg.end:.2f}s, OCR: {texts}")
        else:
            print(f"区间{i+1}: {seg.start:.2f}-{seg.end:.2f}s, 未提取到帧")

    video_processor.close()


if __name__ == "__main__":
    main() 