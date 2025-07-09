"""
VAD+ASR对一个视频文件进行硬字幕提取：

1. 加载视频文件；
2. 提取视频中的音频轨道；
3. 使用语音活动检测（VAD）对音频进行分段，检测出语音区间；
4. 针对每个语音区间，进行切割成单个音频；
5. 对提取的每个音频进行ASR的识别。

"""
from src.extractors import VideoTranscriptionExtractor
from src.core.types import ProcessingConfig


def main():
    video_path = 'sample_data/20250626上海话视频480568590_nb2-1-16.mp4'
    config = ProcessingConfig(ocr_languages=["ch"])
    
    # 使用VideoTranscriptionExtractor进行ASR处理
    processor = VideoTranscriptionExtractor(config)
    result = processor.process_video_asr(video_path, \
                                         asr_model_file="/mnt/sda/ASR/zhanghui/FunASR/inference_model/secondmodel/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-jingzhou")
    
    if result:
        print("ASR处理完成")
    else:
        print("ASR处理失败")


if __name__ == "__main__":
    main() 
