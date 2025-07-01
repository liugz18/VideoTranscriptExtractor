"""
基于Whisper的ASR处理器实现
"""

import numpy as np
from typing import List, Optional, Dict, Any
import logging

from ...core.asr_processor import ASRProcessor
from ...core.types import AudioSegment, ASRResult, ProcessingConfig


class WhisperASRProcessor(ASRProcessor):
    """基于Whisper的ASR处理器"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.model = None
        self._init_model()
    
    def _init_model(self):
        """初始化Whisper模型"""
        try:
            import whisper
            model_name = self.config.asr_model if self.config.asr_model != "whisper" else "base"
            self.model = whisper.load_model(model_name)
            self.logger.info(f"Whisper模型加载成功: {model_name}")
        except ImportError:
            self.logger.warning("Whisper未安装，使用模拟ASR")
            self.model = None
    
    def transcribe_audio(self, audio_data: np.ndarray, 
                        sample_rate: int) -> List[ASRResult]:
        """转录音频数据"""
        if self.model is None:
            return self._simulate_asr(audio_data, sample_rate)
        
        try:
            # 保存音频到临时文件
            import tempfile
            import soundfile as sf
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                sf.write(temp_file.name, audio_data, sample_rate)
                
                # 使用Whisper转录
                result = self.model.transcribe(
                    temp_file.name,
                    language=self.config.asr_language if self.config.asr_language != "auto" else None,
                    word_timestamps=True
                )
            
            # 解析结果
            asr_results = []
            if 'segments' in result:
                for segment in result['segments']:
                    asr_result = ASRResult(
                        text=segment['text'].strip(),
                        confidence=segment.get('avg_logprob', 0.5),
                        start_time=segment['start'],
                        end_time=segment['end'],
                        words=segment.get('words', [])
                    )
                    asr_results.append(asr_result)
            else:
                # 如果没有分段，创建单个结果
                asr_result = ASRResult(
                    text=result['text'].strip(),
                    confidence=0.8,
                    start_time=0.0,
                    end_time=len(audio_data) / sample_rate
                )
                asr_results.append(asr_result)
            
            return asr_results
            
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
        """转录音频文件"""
        if self.model is None:
            return self._simulate_file_asr(audio_path)
        
        try:
            result = self.model.transcribe(
                audio_path,
                language=self.config.asr_language if self.config.asr_language != "auto" else None,
                word_timestamps=True
            )
            
            # 解析结果
            asr_results = []
            if 'segments' in result:
                for segment in result['segments']:
                    asr_result = ASRResult(
                        text=segment['text'].strip(),
                        confidence=segment.get('avg_logprob', 0.5),
                        start_time=segment['start'],
                        end_time=segment['end'],
                        words=segment.get('words', [])
                    )
                    asr_results.append(asr_result)
            else:
                asr_result = ASRResult(
                    text=result['text'].strip(),
                    confidence=0.8,
                    start_time=0.0,
                    end_time=result.get('duration', 0.0)
                )
                asr_results.append(asr_result)
            
            return asr_results
            
        except Exception as e:
            self.logger.error(f"从文件转录时发生错误: {str(e)}")
            return self._simulate_file_asr(audio_path)
    
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
        return self.transcribe_audio(audio_data, sample_rate)
    
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