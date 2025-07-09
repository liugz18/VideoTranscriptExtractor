"""
VAD+OCR对一个视频文件进行硬字幕提取：

1. 加载视频文件；
2. 提取视频中的音频轨道；
3. 使用语音活动检测（VAD）对音频进行分段，检测出语音区间；
4. 每个区间采样多帧做OCR，检测字幕变化；
5. 将所有OCR结果合并，输出字幕文本。
"""
from src.extractors import VideoTranscriptionExtractor
from src.core.types import ProcessingConfig


def main():
    video_path = 'sample_data/20250626上海话视频480568590_nb2-1-16.mp4'
    config = ProcessingConfig(ocr_languages=["ch"])
    
    # 使用VideoTranscriptionExtractor进行OCR处理
    processor = VideoTranscriptionExtractor(config)
    result = processor.process_video_ocr(
        video_path,
        sample_interval=20,  # 每20帧采样一次
        crop_ratio=0.45,     # 裁剪下45%区域
        similarity_threshold=0.6  # 相似度阈值
    )
    
    if result:
        print("OCR处理完成")
    else:
        print("OCR处理失败")


if __name__ == "__main__":
    main() 
