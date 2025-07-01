"""
VideoTranscriptExtractor 基本使用示例
"""

import logging
from pathlib import Path

from src.core import VideoTranscriptExtractor, ProcessingConfig
from src.processors import (
    OpenCVVideoProcessor,
    FFmpegAudioProcessor,
    WebRTCVADProcessor,
    EasyOCRProcessor,
    WhisperASRProcessor,
    SimpleFusionProcessor
)


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_processors():
    """创建处理器实例"""
    # 创建配置
    config = ProcessingConfig(
        vad_threshold=0.5,
        vad_min_duration=0.5,
        vad_max_duration=10.0,
        ocr_confidence_threshold=0.7,
        ocr_languages=["en", "zh"],
        asr_model="base",
        asr_language="auto",
        fusion_weight_ocr=0.6,
        fusion_weight_asr=0.4
    )
    
    # 创建处理器
    video_processor = OpenCVVideoProcessor(config)
    audio_processor = FFmpegAudioProcessor(config)
    vad_detector = WebRTCVADProcessor(config)
    ocr_processor = EasyOCRProcessor(config)
    asr_processor = WhisperASRProcessor(config)
    fusion_processor = SimpleFusionProcessor(config)
    
    return {
        'video_processor': video_processor,
        'audio_processor': audio_processor,
        'vad_detector': vad_detector,
        'ocr_processor': ocr_processor,
        'asr_processor': asr_processor,
        'fusion_processor': fusion_processor,
        'config': config
    }


def main():
    """主函数"""
    setup_logging()
    
    # 创建处理器
    processors = create_processors()
    
    # 创建转录提取器
    extractor = VideoTranscriptExtractor(
        video_processor=processors['video_processor'],
        audio_processor=processors['audio_processor'],
        vad_detector=processors['vad_detector'],
        ocr_processor=processors['ocr_processor'],
        asr_processor=processors['asr_processor'],
        fusion_processor=processors['fusion_processor'],
        config=processors['config']
    )
    
    # 视频文件路径
    video_path = "path/to/your/video.mp4"
    
    # 检查文件是否存在
    if not Path(video_path).exists():
        print(f"视频文件不存在: {video_path}")
        print("请将视频文件路径替换为实际路径")
        return
    
    try:
        # 提取转录
        print("开始提取视频转录...")
        result = extractor.extract(video_path)
        
        # 输出结果
        print("\n=== 转录结果 ===")
        print(f"视频时长: {result.duration:.2f}秒")
        print(f"片段数量: {len(result.segments)}")
        print(f"完整文本: {result.transcript}")
        
        print("\n=== 详细片段 ===")
        for i, segment in enumerate(result.segments):
            print(f"片段 {i+1}:")
            print(f"  时间: {segment.start_time:.2f}s - {segment.end_time:.2f}s")
            print(f"  文本: {segment.text}")
            print(f"  来源: {segment.source}")
            print(f"  置信度: {segment.confidence:.2f}")
            print()
        
        print("\n=== 元数据 ===")
        for key, value in result.metadata.items():
            print(f"{key}: {value}")
        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
    
    finally:
        # 清理资源
        processors['video_processor'].close()


if __name__ == "__main__":
    main() 