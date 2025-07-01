"""
集成测试 - 测试视频转音频、VAD检测、OCR识别的完整流程
"""
import pytest
from src.core import VideoTranscriptExtractor, ProcessingConfig
from src.processors.video.opencv_processor import OpenCVVideoProcessor
from src.processors.audio.ffmpeg_processor import FFmpegAudioProcessor
from src.processors.vad.silero_vad_processor import SileroVADProcessor
from src.processors.ocr.paddleocr_processor import PaddleOCRProcessor


def test_video_to_audio_to_vad_to_ocr():
    """测试视频转音频、VAD检测、OCR识别的完整流程"""
    # 假设测试视频存在
    video_path = 'sample_data/20250626上海话视频480568590_nb2-1-16.mp4'
    
    # 创建配置
    config = ProcessingConfig(
        vad_threshold=0.5,
        vad_min_duration=0.5,
        vad_max_duration=10.0,
        ocr_confidence_threshold=0.7,
        ocr_languages=["ch"],
        asr_model="base",
        asr_language="auto",
        fusion_weight_ocr=0.6,
        fusion_weight_asr=0.4
    )
    
    # 创建处理器
    video_processor = OpenCVVideoProcessor(config)
    audio_processor = FFmpegAudioProcessor(config)
    vad_detector = SileroVADProcessor(config)
    ocr_processor = PaddleOCRProcessor(config)
    
    try:
        # 1. 加载视频
        print("1. 加载视频...")
        success = video_processor.load_video(video_path)
        assert success, "视频加载失败"
        video_info = video_processor.get_video_info()
        print(f"视频信息: {video_info}")
        
        # 2. 提取音频
        print("2. 提取音频...")
        audio_path = audio_processor.extract_audio_from_video(video_path)
        assert audio_path is not None, "音频提取失败"
        print(f"音频文件: {audio_path}")
        
        # 3. VAD检测语音区间
        print("3. VAD检测语音区间...")
        voice_segments = vad_detector.detect_voice_segments_from_file(audio_path)
        print(f"检测到 {len(voice_segments)} 个语音区间")
        for i, segment in enumerate(voice_segments[:3]):  # 只打印前3个
            print(f"  区间 {i+1}: {segment.start:.2f}s - {segment.end:.2f}s")
        
        # 4. 提取视频帧进行OCR
        print("4. 提取视频帧进行OCR...")
        if voice_segments:
            # 取第一个语音区间进行测试
            test_segment = voice_segments[0]
            frames = video_processor.extract_frames([(test_segment.start, test_segment.end)])
            
            if frames and frames[0]:
                print(f"提取到 {len(frames[0])} 帧")
                # 对前几帧进行OCR
                for i, frame in enumerate(frames[0][:3]):  # 只处理前3帧
                    print(f"  处理帧 {i+1} (时间: {frame.timestamp:.2f}s)")
                    ocr_results = ocr_processor.recognize_text(frame.frame_data)
                    print(f"    OCR结果: {[r.text for r in ocr_results]}")
            else:
                print("未提取到视频帧")
        else:
            print("未检测到语音区间，跳过OCR测试")
        
        print("集成测试完成！")
        
    except Exception as e:
        pytest.fail(f"集成测试失败: {str(e)}")
    
    finally:
        # 清理资源
        video_processor.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 