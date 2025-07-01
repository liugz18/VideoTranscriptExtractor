"""
PaddleOCRProcessor 测试
"""
from src.processors.ocr.paddleocr_processor import PaddleOCRProcessor
from src.core.types import OCRResult

def test_paddleocr_on_real_image():
    # 使用真实图片 sample_data/20250701OCR测试.png
    img_path = 'sample_data/20250701OCR测试.png'
    processor = PaddleOCRProcessor()
    results = processor.recognize_text_from_file(img_path)
    print("OCR结果:", [r.text for r in results])
    # 占位符GT
    gt_text = "GT_PLACEHOLDER"
    # 断言至少有结果，且第一个结果与GT占位符比较（实际使用时替换GT）
    assert len(results) > 0
    # assert results[0].text == gt_text  # 实际使用时取消注释并替换GT

if __name__ == "main":
    test_paddleocr_on_real_image()