# VideoTranscriptExtractor

一个用于从视频中提取语音转录的开源工具，通过融合OCR（硬字幕识别）和ASR（语音识别）技术，提供高质量的转录结果。

## 功能特性

- 🎥 支持多种视频格式输入
- 🔊 基于VAD（Voice Activity Detection）的语音区间检测
- 📝 OCR识别视频中的硬字幕
- 🎤 ASR语音转文字
- 🔄 智能融合OCR和ASR结果
- 🎯 高精度转录输出

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

```python
from video_transcript_extractor import VideoTranscriptExtractor

extractor = VideoTranscriptExtractor()
result = extractor.extract("path/to/video.mp4")
print(result.transcript)
```

## 项目结构

```
VideoTranscriptExtractor/
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── video_processor.py      # 视频处理接口
│   │   ├── audio_processor.py      # 音频处理接口
│   │   ├── vad_detector.py         # VAD检测接口
│   │   ├── ocr_processor.py        # OCR处理接口
│   │   ├── asr_processor.py        # ASR处理接口
│   │   └── fusion_processor.py     # 结果融合接口
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── video/
│   │   ├── audio/
│   │   ├── vad/
│   │   ├── ocr/
│   │   ├── asr/
│   │   └── fusion/
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── tests/
├── examples/
├── requirements.txt
└── setup.py
```

## 许可证

MIT License 