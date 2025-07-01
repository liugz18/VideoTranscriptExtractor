"""
PaddleOCRProcessor 测试
"""
import numpy as np
import pytest
from src.processors.ocr.paddleocr_processor import PaddleOCRProcessor
from src.core.types import VideoFrame, OCRResult

@pytest.fixture
def example_image(tmp_path):
    # 生成一个简单的黑底白字图片
    import cv2
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    cv2.putText(img, "测试", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
    img_path = tmp_path / "test.png"
    cv2.imwrite(str(img_path), img)
    return str(img_path), img

def test_recognize_text_from_file(example_image):
    img_path, _ = example_image
    processor = PaddleOCRProcessor()
    results = processor.recognize_text_from_file(img_path)
    assert isinstance(results, list)
    assert all(isinstance(r, OCRResult) for r in results)
    assert any(r.text for r in results)

def test_recognize_text_from_frame(example_image):
    _, img = example_image
    processor = PaddleOCRProcessor()
    results = processor.recognize_text(img)
    assert isinstance(results, list)
    assert all(isinstance(r, OCRResult) for r in results)
    assert any(r.text == "测试" for r in results)
    