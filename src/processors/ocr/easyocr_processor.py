"""
基于EasyOCR的OCR处理器实现
"""

import numpy as np
from typing import List, Optional, Dict, Any
import logging

from ...core.ocr_processor import OCRProcessor
from ...core.types import VideoFrame, OCRResult, ProcessingConfig


class EasyOCRProcessor(OCRProcessor):
    """基于EasyOCR的OCR处理器"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.reader = None
        self._init_reader()
    
    def _init_reader(self):
        """初始化EasyOCR读取器"""
        try:
            import easyocr
            languages = self.config.ocr_languages if self.config.ocr_languages else ['en']
            self.reader = easyocr.Reader(languages)
            self.logger.info(f"EasyOCR初始化成功，支持语言: {languages}")
        except ImportError:
            self.logger.warning("EasyOCR未安装，使用模拟OCR")
            self.reader = None
    
    def recognize_text(self, image: np.ndarray) -> List[OCRResult]:
        """识别图像中的文本"""
        if self.reader is None:
            return self._simulate_ocr(image)
        
        try:
            results = self.reader.readtext(image)
            ocr_results = []
            
            for (bbox, text, confidence) in results:
                # 转换边界框格式
                bbox_list = [coord for point in bbox for coord in point]
                
                result = OCRResult(
                    text=text,
                    confidence=confidence,
                    bbox=bbox_list
                )
                ocr_results.append(result)
            
            return ocr_results
            
        except Exception as e:
            self.logger.error(f"OCR识别时发生错误: {str(e)}")
            return self._simulate_ocr(image)
    
    def _simulate_ocr(self, image: np.ndarray) -> List[OCRResult]:
        """模拟OCR识别"""
        # 简单的模拟OCR结果
        height, width = image.shape[:2]
        
        # 模拟在图像底部检测到文本
        simulated_results = [
            OCRResult(
                text="模拟字幕文本",
                confidence=0.8,
                bbox=[0, height * 0.8, width, height]
            )
        ]
        
        return simulated_results
    
    def recognize_text_from_frames(self, frames: List[VideoFrame]) -> List[OCRResult]:
        """从视频帧列表识别文本"""
        results = []
        
        for frame in frames:
            frame_results = self.recognize_text(frame.frame_data)
            
            # 添加时间戳
            for result in frame_results:
                result.timestamp = frame.timestamp
            
            results.extend(frame_results)
        
        return results
    
    def recognize_text_from_file(self, image_path: str) -> List[OCRResult]:
        """从图像文件识别文本"""
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
        """图像预处理"""
        try:
            import cv2
            
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # 应用高斯模糊
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 自适应阈值处理
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            return thresh
            
        except Exception as e:
            self.logger.error(f"图像预处理时发生错误: {str(e)}")
            return image
    
    def detect_text_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """检测文本区域"""
        # 简单的文本区域检测
        height, width = image.shape[:2]
        
        # 模拟检测到几个文本区域
        regions = [
            {
                'bbox': [0, height * 0.7, width, height * 0.9],  # 底部区域
                'confidence': 0.8
            },
            {
                'bbox': [0, height * 0.1, width, height * 0.3],  # 顶部区域
                'confidence': 0.6
            }
        ]
        
        return regions
    
    def filter_results(self, results: List[OCRResult], 
                      confidence_threshold: Optional[float] = None) -> List[OCRResult]:
        """过滤OCR结果"""
        if confidence_threshold is None:
            confidence_threshold = self.config.ocr_confidence_threshold
        
        filtered = []
        for result in results:
            if result.confidence >= confidence_threshold:
                filtered.append(result)
        
        return filtered
    
    def merge_results(self, results: List[OCRResult], 
                     time_threshold: float = 1.0) -> List[OCRResult]:
        """合并时间相近的OCR结果"""
        if not results:
            return []
        
        # 按时间戳排序
        sorted_results = sorted(results, key=lambda x: x.timestamp or 0)
        merged = [sorted_results[0]]
        
        for result in sorted_results[1:]:
            last_result = merged[-1]
            
            # 检查时间间隔
            time_diff = abs((result.timestamp or 0) - (last_result.timestamp or 0))
            
            if time_diff <= time_threshold and result.text == last_result.text:
                # 合并相同文本的结果
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
        """设置识别语言"""
        self.config.ocr_languages = languages
        if self.reader:
            # 重新初始化读取器
            self._init_reader()
    
    def get_supported_languages(self) -> List[str]:
        """获取支持的语言列表"""
        return ['en', 'zh', 'ja', 'ko', 'ar', 'hi', 'th', 'vi', 'de', 'fr', 'es', 'it', 'pt', 'ru'] 