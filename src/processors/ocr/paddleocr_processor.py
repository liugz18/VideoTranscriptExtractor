"""
基于PaddleOCR的OCR处理器实现
"""

import numpy as np
from typing import List, Optional, Dict, Any
import logging

from ...core.ocr_processor import OCRProcessor
from ...core.types import VideoFrame, OCRResult, ProcessingConfig

class PaddleOCRProcessor(OCRProcessor):
    """基于PaddleOCR的OCR处理器"""
    def __init__(self, config: Optional[ProcessingConfig] = None):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.ocr = None
        self._init_ocr()

    def _init_ocr(self):
        try:
            from paddleocr import PaddleOCR
            lang = self.config.ocr_languages[0] if self.config.ocr_languages else "ch"
            self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)
        except ImportError:
            self.logger.warning("PaddleOCR未安装，使用模拟OCR")
            self.ocr = None

    def recognize_text(self, image: np.ndarray) -> List[OCRResult]:
        if self.ocr is None:
            return self._simulate_ocr(image)
        try:
            results = self.ocr.ocr(image)[0]
            ocr_results = []
            for i in range(len(results['rec_texts'])):
                ocr_results.append(OCRResult(results['rec_texts'][i], results['rec_scores'][i]))
            return ocr_results
        except Exception as e:
            self.logger.error(f"PaddleOCR识别时发生错误: {str(e)}")
            return self._simulate_ocr(image)

    def _simulate_ocr(self, image: np.ndarray) -> List[OCRResult]:
        height, width = image.shape[:2]
        return [OCRResult(text="模拟字幕文本", confidence=0.8, bbox=[0, height * 0.8, width, height])]

    def recognize_text_from_frames(self, frames: List[VideoFrame], sample_interval: int = 5, crop_ratio: float = 0.45, similarity_threshold: float = 0.6) -> List[OCRResult]:
        """
        对帧序列以sample_interval为步长采样，检测OCR结果变化，输出区间内所有不同的OCR结果。
        Args:
            frames: 视频帧列表
            sample_interval: 采样间隔（帧数），默认每5帧采样一次
            crop_ratio: 裁剪比例，默认0.45表示裁剪掉上1/2的图片
            similarity_threshold: 相似度阈值，默认0.6，超过此阈值认为字幕相同
        Returns:
            List[OCRResult]: 不同的OCR结果列表
        """
        results = []
        last_texts = None
        for idx in range(0, len(frames), sample_interval):
            frame = frames[idx]
            
            # 裁剪出下crop_ratio比例的图片
            height, width = frame.frame_data.shape[:2]
            cropped_frame = frame.frame_data[int((1-crop_ratio)*height):height, :]
            
            frame_results = self.recognize_text(cropped_frame)
            # 只关注文本内容
            texts = tuple([r.text for r in frame_results if r.text])
            if not texts:
                continue
            
            # 使用模糊比对判断字幕是否变化
            if last_texts is None or not self._is_similar_texts(texts, last_texts, similarity_threshold):
                # 记录变化的OCR结果
                for r in frame_results:
                    r.timestamp = frame.timestamp
                results.extend(frame_results)
                last_texts = texts
        return results

    def _is_similar_texts(self, texts1: tuple, texts2: tuple, threshold: float) -> bool:
        """
        判断两组文本是否相似
        Args:
            texts1: 第一组文本
            texts2: 第二组文本
            threshold: 相似度阈值
        Returns:
            bool: 是否相似
        """
        if not texts1 or not texts2:
            return False
        
        # 计算文本相似度（简单实现，可扩展为更复杂的算法）
        text1 = " ".join(texts1)
        text2 = " ".join(texts2)
        
        # 使用编辑距离计算相似度
        distance = self._levenshtein_distance(text1, text2)
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return True
        
        similarity = 1 - (distance / max_len)
        # print(text1, text2, distance, similarity)
        return similarity >= threshold

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        计算两个字符串的编辑距离
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

    def recognize_text_from_file(self, image_path: str) -> List[OCRResult]:
        try:
            import cv2
            image = cv2.imread(image_path)
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return self.recognize_text(image_rgb)
            else:
                self.logger.error(f"无法读取图像文件: {image_path}")
                return []
        except Exception as e:
            self.logger.error(f"从文件识别文本时发生错误: {str(e)}")
            return []

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        try:
            import cv2
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            return thresh
        except Exception as e:
            self.logger.error(f"图像预处理时发生错误: {str(e)}")
            return image

    def detect_text_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        height, width = image.shape[:2]
        regions = [
            {'bbox': [0, height * 0.7, width, height * 0.9], 'confidence': 0.8},
            {'bbox': [0, height * 0.1, width, height * 0.3], 'confidence': 0.6}
        ]
        return regions

    def filter_results(self, results: List[OCRResult], confidence_threshold: Optional[float] = None) -> List[OCRResult]:
        if confidence_threshold is None:
            confidence_threshold = self.config.ocr_confidence_threshold
        return [r for r in results if r.confidence >= confidence_threshold]

    def merge_results(self, results: List[OCRResult], time_threshold: float = 1.0) -> List[OCRResult]:
        if not results:
            return []
        sorted_results = sorted(results, key=lambda x: x.timestamp or 0)
        merged = [sorted_results[0]]
        for result in sorted_results[1:]:
            last_result = merged[-1]
            time_diff = abs((result.timestamp or 0) - (last_result.timestamp or 0))
            if time_diff <= time_threshold and result.text == last_result.text:
                merged[-1] = OCRResult(
                    text=result.text,
                    confidence=max(result.confidence, last_result.confidence),
                    bbox=result.bbox,
                    timestamp=result.timestamp
                )
            else:
                merged.append(result)
        return merged

    def set_languages(self, languages: List[str]):
        self.config.ocr_languages = languages
        self._init_ocr()

    def get_supported_languages(self) -> List[str]:
        return ['ch', 'en', 'fr', 'german', 'korean', 'japan', 'ru', 'it', 'es', 'pt', 'ar', 'hi', 'th', 'vi'] 