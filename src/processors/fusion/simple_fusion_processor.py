"""
简单的融合处理器实现
"""

import numpy as np
from typing import List, Optional, Dict, Any
import logging
from difflib import SequenceMatcher

from ...core.fusion_processor import FusionProcessor
from ...core.types import OCRResult, ASRResult, TranscriptionSegment, ProcessingConfig


class SimpleFusionProcessor(FusionProcessor):
    """简单的融合处理器"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.fusion_strategy = "confidence_weighted"
    
    def fuse_results(self, ocr_results: List[OCRResult], 
                    asr_results: List[ASRResult]) -> List[TranscriptionSegment]:
        """融合OCR和ASR结果"""
        # 对齐结果
        aligned_results = self.align_results(ocr_results, asr_results)
        
        segments = []
        for aligned in aligned_results:
            ocr_result = aligned.get('ocr')
            asr_result = aligned.get('asr')
            
            if ocr_result and asr_result:
                # 两个结果都存在，进行融合
                best_result = self.select_best_result(ocr_result, asr_result)
                segment = TranscriptionSegment(
                    start_time=aligned['start_time'],
                    end_time=aligned['end_time'],
                    text=best_result['text'],
                    source=best_result['source'],
                    confidence=best_result['confidence']
                )
            elif ocr_result:
                # 只有OCR结果
                segment = TranscriptionSegment(
                    start_time=aligned['start_time'],
                    end_time=aligned['end_time'],
                    text=ocr_result.text,
                    source='ocr',
                    confidence=ocr_result.confidence
                )
            elif asr_result:
                # 只有ASR结果
                segment = TranscriptionSegment(
                    start_time=aligned['start_time'],
                    end_time=aligned['end_time'],
                    text=asr_result.text,
                    source='asr',
                    confidence=asr_result.confidence
                )
            else:
                continue
            
            segments.append(segment)
        
        return segments
    
    def align_results(self, ocr_results: List[OCRResult], 
                     asr_results: List[ASRResult]) -> List[Dict[str, Any]]:
        """对齐OCR和ASR结果"""
        aligned = []
        
        # 按时间排序
        ocr_sorted = sorted(ocr_results, key=lambda x: x.timestamp or 0)
        asr_sorted = sorted(asr_results, key=lambda x: x.start_time)
        
        # 简单的基于时间的对齐
        for ocr_result in ocr_sorted:
            ocr_time = ocr_result.timestamp or 0
            
            # 找到最接近的ASR结果
            best_asr = None
            best_time_diff = float('inf')
            
            for asr_result in asr_sorted:
                asr_time = asr_result.start_time
                time_diff = abs(ocr_time - asr_time)
                
                if time_diff < best_time_diff and time_diff <= 2.0:  # 2秒内
                    best_time_diff = time_diff
                    best_asr = asr_result
            
            aligned.append({
                'start_time': ocr_time,
                'end_time': ocr_time + 1.0,  # 假设1秒持续时间
                'ocr': ocr_result,
                'asr': best_asr
            })
        
        # 添加没有对应OCR的ASR结果
        for asr_result in asr_sorted:
            asr_time = asr_result.start_time
            
            # 检查是否已经有对应的OCR结果
            has_ocr = any(
                abs(aligned_item['start_time'] - asr_time) <= 2.0 
                for aligned_item in aligned
            )
            
            if not has_ocr:
                aligned.append({
                    'start_time': asr_time,
                    'end_time': asr_result.end_time,
                    'ocr': None,
                    'asr': asr_result
                })
        
        # 按时间排序
        aligned.sort(key=lambda x: x['start_time'])
        
        return aligned
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度"""
        if not text1 or not text2:
            return 0.0
        
        # 使用序列匹配器计算相似度
        similarity = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        return similarity
    
    def merge_similar_texts(self, texts: List[str], 
                          similarity_threshold: float = 0.8) -> List[str]:
        """合并相似的文本"""
        if not texts:
            return []
        
        merged = []
        used_indices = set()
        
        for i, text1 in enumerate(texts):
            if i in used_indices:
                continue
            
            similar_texts = [text1]
            used_indices.add(i)
            
            for j, text2 in enumerate(texts[i+1:], i+1):
                if j in used_indices:
                    continue
                
                if self.calculate_similarity(text1, text2) >= similarity_threshold:
                    similar_texts.append(text2)
                    used_indices.add(j)
            
            # 选择最长的文本作为代表
            best_text = max(similar_texts, key=len)
            merged.append(best_text)
        
        return merged
    
    def select_best_result(self, ocr_result: OCRResult, 
                          asr_result: ASRResult) -> Dict[str, Any]:
        """选择最佳结果"""
        if self.fusion_strategy == "confidence_weighted":
            # 基于置信度加权
            ocr_weight = self.config.fusion_weight_ocr
            asr_weight = self.config.fusion_weight_asr
            
            ocr_score = ocr_result.confidence * ocr_weight
            asr_score = asr_result.confidence * asr_weight
            
            if ocr_score > asr_score:
                return {
                    'text': ocr_result.text,
                    'source': 'ocr',
                    'confidence': ocr_result.confidence
                }
            else:
                return {
                    'text': asr_result.text,
                    'source': 'asr',
                    'confidence': asr_result.confidence
                }
        
        elif self.fusion_strategy == "similarity_based":
            # 基于相似度
            similarity = self.calculate_similarity(ocr_result.text, asr_result.text)
            
            if similarity > 0.7:
                # 文本相似，选择置信度更高的
                if ocr_result.confidence > asr_result.confidence:
                    return {
                        'text': ocr_result.text,
                        'source': 'ocr',
                        'confidence': ocr_result.confidence
                    }
                else:
                    return {
                        'text': asr_result.text,
                        'source': 'asr',
                        'confidence': asr_result.confidence
                    }
            else:
                # 文本不相似，选择置信度更高的
                if ocr_result.confidence > asr_result.confidence:
                    return {
                        'text': ocr_result.text,
                        'source': 'ocr',
                        'confidence': ocr_result.confidence
                    }
                else:
                    return {
                        'text': asr_result.text,
                        'source': 'asr',
                        'confidence': asr_result.confidence
                    }
        
        else:
            # 默认选择置信度更高的
            if ocr_result.confidence > asr_result.confidence:
                return {
                    'text': ocr_result.text,
                    'source': 'ocr',
                    'confidence': ocr_result.confidence
                }
            else:
                return {
                    'text': asr_result.text,
                    'source': 'asr',
                    'confidence': asr_result.confidence
                }
    
    def interpolate_missing_segments(self, segments: List[TranscriptionSegment], 
                                   total_duration: float) -> List[TranscriptionSegment]:
        """插值缺失的转录片段"""
        if not segments:
            return []
        
        # 按时间排序
        sorted_segments = sorted(segments, key=lambda x: x.start_time)
        interpolated = []
        
        # 添加开始到第一个片段的间隔
        if sorted_segments[0].start_time > 0:
            interpolated.append(TranscriptionSegment(
                start_time=0.0,
                end_time=sorted_segments[0].start_time,
                text="[静音]",
                source='interpolation',
                confidence=1.0
            ))
        
        # 处理片段之间的间隔
        for i in range(len(sorted_segments) - 1):
            current = sorted_segments[i]
            next_segment = sorted_segments[i + 1]
            
            interpolated.append(current)
            
            # 检查是否有间隔
            if next_segment.start_time - current.end_time > 0.5:  # 0.5秒以上间隔
                interpolated.append(TranscriptionSegment(
                    start_time=current.end_time,
                    end_time=next_segment.start_time,
                    text="[静音]",
                    source='interpolation',
                    confidence=1.0
                ))
        
        # 添加最后一个片段
        if sorted_segments:
            interpolated.append(sorted_segments[-1])
        
        # 添加最后一个片段到结束的间隔
        if interpolated and interpolated[-1].end_time < total_duration:
            interpolated.append(TranscriptionSegment(
                start_time=interpolated[-1].end_time,
                end_time=total_duration,
                text="[静音]",
                source='interpolation',
                confidence=1.0
            ))
        
        return interpolated
    
    def smooth_segments(self, segments: List[TranscriptionSegment], 
                       window_size: float = 1.0) -> List[TranscriptionSegment]:
        """平滑转录片段"""
        if not segments:
            return []
        
        # 按时间排序
        sorted_segments = sorted(segments, key=lambda x: x.start_time)
        smoothed = []
        
        for i, segment in enumerate(sorted_segments):
            # 收集窗口内的片段
            window_start = segment.start_time - window_size / 2
            window_end = segment.end_time + window_size / 2
            
            window_segments = []
            for j, other_segment in enumerate(sorted_segments):
                if (other_segment.start_time <= window_end and 
                    other_segment.end_time >= window_start):
                    window_segments.append(other_segment)
            
            # 如果窗口内有多个相似文本，选择最频繁的
            if len(window_segments) > 1:
                text_counts = {}
                for seg in window_segments:
                    text = seg.text.strip()
                    if text and text != "[静音]":
                        text_counts[text] = text_counts.get(text, 0) + 1
                
                if text_counts:
                    most_common_text = max(text_counts, key=text_counts.get)
                    smoothed.append(TranscriptionSegment(
                        start_time=segment.start_time,
                        end_time=segment.end_time,
                        text=most_common_text,
                        source='fusion',
                        confidence=segment.confidence
                    ))
                else:
                    smoothed.append(segment)
            else:
                smoothed.append(segment)
        
        return smoothed
    
    def set_fusion_weights(self, ocr_weight: float, asr_weight: float):
        """设置融合权重"""
        self.config.fusion_weight_ocr = ocr_weight
        self.config.fusion_weight_asr = asr_weight
    
    def get_fusion_strategy(self) -> str:
        """获取融合策略"""
        return self.fusion_strategy
    
    def set_fusion_strategy(self, strategy: str):
        """设置融合策略"""
        valid_strategies = ["confidence_weighted", "similarity_based", "confidence_only"]
        if strategy in valid_strategies:
            self.fusion_strategy = strategy
        else:
            self.logger.warning(f"无效的融合策略: {strategy}") 