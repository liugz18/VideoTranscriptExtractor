"""
基于funasr的ASR处理器实现
"""

import numpy as np
from typing import List, Optional, Dict, Any
import logging

from ...core.asr_processor import ASRProcessor
from ...core.types import AudioSegment, ASRResult, ProcessingConfig


class ParaformerASRProcessor(ASRProcessor):
    """基于Whisper的ASR处理器"""

    def __init__(self, config: Optional[ProcessingConfig] = None,model_file: Optional[str] = None):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.model_file = model_file
        print(f"self.model_file:{model_file}")
        self._init_model()

    def _init_model(self):
        """初始化Whisper模型"""
        try:
            # import whisper
            # model_name = self.config.asr_model if self.config.asr_model != "whisper" else "base"
            # self.model = whisper.load_model(model_name)
            # self.logger.info(f"Whisper模型加载成功: {model_name}")
            if not self.model_file:
                # 如果没有传入 model_file，这里可以提供一个默认值或报错
                self.logger.warning("未传入 model_file，将使用默认模型路径")
            from funasr import AutoModel
            self.model = AutoModel(
                model=self.model_file,
                device="cuda"
            )
            self.logger.info("FunASR 模型加载成功")
        # except ImportError:
        #     self.logger.warning("Whisper未安装，使用模拟ASR")
        #     self.model = None
        except ImportError:
            self.logger.error("未安装 FunASR，请先安装")
            self.model = None
        except Exception as e:
            self.logger.error(f"FunASR 模型加载失败: {e}")
            self.model = None
    def transcribe_audio(self, audio_data: np.ndarray,
                        sample_rate: int) -> List[ASRResult]:
        """转录音频数据"""
        if self.model is None:
            return self._simulate_asr(audio_data, sample_rate)

        try:
            if self.model:
                self.logger.info("ASR 模型已准备好")
            else:
                self.logger.warning("ASR 模型未加载，使用模拟 ASR")
            
            audio_data = audio_data.astype(np.float32) / 32768.0
            res = self.model.generate(input=audio_data,
                                 batch_size_s=300,
                                 output_dir="./output"
                                 )
            print(f"res:{res}")
#            return res[0]["text"].replace(" ", "")
            text = res[0].get("text", "").replace(" ", "")
            duration = len(audio_data) / sample_rate
            # 返回统一格式 List[ASRResult]
            return [
                ASRResult(
                    text=text,
                    confidence=1.0,  # 模型没有返回confidence时，默认1.0或其他值
                    start_time=0.0,
                    end_time=duration
                )
            ]
        except Exception as e:
            self.logger.error(f"ASR转录时发生错误: {str(e)}")
            return self._simulate_asr(audio_data, sample_rate)

    def _simulate_asr(self, audio_data: np.ndarray, sample_rate: int) -> List[ASRResult]:
        """模拟ASR转录"""
        duration = len(audio_data) / sample_rate

        # 模拟转录结果
        simulated_results = [
            ASRResult(
                text="这是模拟的语音识别结果",
                confidence=0.8,
                start_time=0.0,
                end_time=duration
            )
        ]

        return simulated_results

    def transcribe_audio_file(self, audio_path: str) -> List[ASRResult]:
        pass

    def _simulate_file_asr(self, audio_path: str) -> List[ASRResult]:
        """模拟文件ASR转录"""
        return [
            ASRResult(
                text="模拟的音频文件转录结果",
                confidence=0.8,
                start_time=0.0,
                end_time=30.0
            )
        ]

    def transcribe_audio_segments(self, audio_segments: List[AudioSegment]) -> List[ASRResult]:
        """转录音频片段列表"""
        results = []

        for segment in audio_segments:
            segment_results = self.transcribe_audio(
                segment.audio_data, segment.sample_rate
            )

            # 调整时间戳
            for result in segment_results:
                adjusted_result = ASRResult(
                    text=result.text,
                    confidence=result.confidence,
                    start_time=segment.start_time + result.start_time,
                    end_time=segment.start_time + result.end_time,
                    words=result.words
                )
                results.append(adjusted_result)

        return results

    def transcribe_with_timestamps(self, audio_data: np.ndarray,
                                 sample_rate: int) -> List[ASRResult]:
        """带时间戳的转录"""
        return self.transcribe_audio(audio_data)

    def detect_language(self, audio_data: np.ndarray,
                       sample_rate: int) -> str:
        """检测音频语言"""
        if self.model is None:
            return "en"  # 默认英语

        try:
            import tempfile
            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                sf.write(temp_file.name, audio_data, sample_rate)
                result = self.model.detect_language(temp_file.name)
                return result

        except Exception as e:
            self.logger.error(f"语言检测时发生错误: {str(e)}")
            return "en"

    def set_language(self, language: str):
        """设置识别语言"""
        self.config.asr_language = language

    def get_supported_languages(self) -> List[str]:
        """获取支持的语言列表"""
        return [
            "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"
        ]

    def preprocess_audio(self, audio_data: np.ndarray,
                        sample_rate: int) -> np.ndarray:
        """音频预处理"""
        # Whisper通常需要16kHz采样率
        if sample_rate != 16000:
            audio_data = self.resample_audio(audio_data, sample_rate, 16000)

        # 归一化
        audio_data = self.normalize_audio(audio_data)

        return audio_data

    def filter_results(self, results: List[ASRResult],
                      confidence_threshold: Optional[float] = None) -> List[ASRResult]:
        """过滤ASR结果"""
        if confidence_threshold is None:
            confidence_threshold = 0.5  # 默认阈值

        filtered = []
        for result in results:
            if result.confidence >= confidence_threshold:
                filtered.append(result)

        return filtered
    
    def load_model(self, model_name: Optional[str] = None):
        """加载ASR模型"""
        if model_name:
            self.config.asr_model = model_name
        self._init_model()
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_name": self.config.asr_model,
            "language": self.config.asr_language,
            "is_loaded": self.model is not None
        } 
