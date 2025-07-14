"""
视频处理工具模块
封装常用的视频处理流程，提供可复用的功能
"""

import os
from typing import Optional, List, Dict, Any
from pathlib import Path

from ..processors.video.opencv_processor import OpenCVVideoProcessor
from ..processors.audio.ffmpeg_processor import FFmpegAudioProcessor
from ..processors.vad.silero_vad_processor import SileroVADProcessor
from ..processors.asr.paraformer_processor import ParaformerASRProcessor
from ..processors.ocr.paddleocr_processor import PaddleOCRProcessor
from ..core.types import ProcessingConfig, TranscriptionResult, TranscriptionSegment


class VideoTranscriptionExtractor:
    """视频转录提取工具类"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None,
                 video_processor: Optional[OpenCVVideoProcessor] = None,
                 audio_processor: Optional[FFmpegAudioProcessor] = None,
                 vad_processor: Optional[SileroVADProcessor] = None):
        """
        初始化视频转录提取器
        
        Args:
            config: 处理配置，如果为None则使用默认配置
            video_processor: 视频处理器，如果为None则使用OpenCVVideoProcessor
            audio_processor: 音频处理器，如果为None则使用FFmpegAudioProcessor
            vad_processor: VAD处理器，如果为None则使用SileroVADProcessor
        """
        self.config = config or ProcessingConfig(ocr_languages=["ch"])
        self.video_processor = video_processor or OpenCVVideoProcessor(self.config)
        self.audio_processor = audio_processor or FFmpegAudioProcessor(self.config)
        self.vad_processor = vad_processor or SileroVADProcessor(self.config)
        
    def process_video_asr(self, video_path: str, output_dir: str = "output", 
                         asr_model_file: Optional[str] = None) -> Optional[TranscriptionResult]:
        """
        对视频进行VAD+ASR处理
        
        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
            asr_model_file: ASR模型文件路径，如果为None则使用默认模型
            
        Returns:
            TranscriptionResult: 转录结果，失败时返回None
        """
        try:
            print(f"开始处理视频ASR: {video_path}")
            
            # 1. 加载视频
            print("加载视频...")
            if not self.video_processor.load_video(video_path):
                print("视频加载失败")
                return None
            
            # 2. 提取音频
            print("提取音频...")
            audio_path = self.audio_processor.extract_audio_from_video(video_path)
            if audio_path is None:
                print("音频提取失败")
                return None
            
            # 3. VAD检测区间
            print("VAD检测...")
            segments = self.vad_processor.detect_voice_segments_from_file(audio_path)
            print(f"检测到 {len(segments)} 个语音区间")
            time_tuples = [(segment.start, segment.end) for segment in segments]
            
            # 4. 切割音频
            print("按照VAD切割音频...")
            audio_segments = self.audio_processor.extract_audio_segments(time_tuples, audio_path)
            
            # 5. ASR识别
            asr_processor = ParaformerASRProcessor(
                self.config, 
                model_file=asr_model_file
            )
            
            all_segments = []
            for i, seg in enumerate(audio_segments):
                print(f"处理音频段 {i+1}/{len(audio_segments)}")
                if seg is not None:
                    asr_results = asr_processor.transcribe_audio(seg.audio_data, sample_rate=16000)
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
            
            # 6. 保存结果
            if all_segments:
                video_duration = self.video_processor.get_video_info()['duration']
                transcription_result = TranscriptionResult(
                    segments=all_segments,
                    duration=video_duration,
                    metadata={
                        "source": "asr",
                        "audio_path": audio_path,
                        "total_segments": len(all_segments)
                    }
                )
                
                # 保存到文件
                os.makedirs(output_dir, exist_ok=True)
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                output_path = os.path.join(output_dir, f"{video_name}_transcription[ASR].json")
                transcription_result.save_to_file(output_path, format="json")
                print(f"ASR转录结果已保存到: {output_path}")
                
                return transcription_result
            else:
                print("未检测到任何ASR结果")
                return None
                
        except Exception as e:
            print(f"ASR处理时发生错误: {str(e)}")
            return None
        finally:
            self.video_processor.close()
    
    def process_video_ocr(self, video_path: str, output_dir: str = "output",
                         sample_interval: int = 20, crop_ratio: float = 0.45,
                         similarity_threshold: float = 0.8) -> Optional[TranscriptionResult]:
        """
        对视频进行VAD+OCR处理
        
        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
            sample_interval: 采样间隔（帧数）
            crop_ratio: 裁剪比例
            similarity_threshold: 相似度阈值
            
        Returns:
            TranscriptionResult: 转录结果，失败时返回None
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(output_dir, f"{video_name}_transcription[OCR].json")
            if os.path.isfile(output_path):
                print(f"OCR 结果文件已存在，跳过处理: {output_path}")
                return None

            print(f"开始处理视频OCR: {video_path}")
            
            # 1. 加载视频
            print("加载视频...")
            if not self.video_processor.load_video(video_path):
                print("视频加载失败")
                return None
            
            # 2. 提取音频
            print("提取音频...")
            audio_path = self.audio_processor.extract_audio_from_video(video_path)
            if audio_path is None:
                print("音频提取失败")
                return None
            
            # 3. VAD检测区间
            print("VAD检测...")
            segments = self.vad_processor.detect_voice_segments_from_file(audio_path)
            print(f"检测到 {len(segments)} 个语音区间")
            
            # 4. 每个区间采样多帧做OCR
            ocr_processor = PaddleOCRProcessor(self.config)
            all_segments = []
            
            for i, seg in enumerate(segments):
                frames_list = self.video_processor.extract_frames([(seg.start, seg.end)])
                frames = frames_list[0] if frames_list else []
                if frames:
                    ocr_results = ocr_processor.recognize_text_from_frames(
                        frames, 
                        sample_interval=sample_interval,
                        crop_ratio=crop_ratio,
                        similarity_threshold=similarity_threshold
                    )
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
            
            # 5. 保存结果
            if all_segments:
                video_duration = self.video_processor.get_video_info()['duration']
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
                
                # 保存到文件
                
                
                transcription_result.save_to_file(output_path, format="json")
                print(f"OCR转录结果已保存到: {output_path}")
                
                return transcription_result
            else:
                print("未检测到任何OCR结果")
                return None
                
        except Exception as e:
            print(f"OCR处理时发生错误: {str(e)}")
            return None
        finally:
            self.video_processor.close() 