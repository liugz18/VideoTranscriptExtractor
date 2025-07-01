"""
PaddleOCRProcessor OCR调用示例
"""
from src.processors.ocr.paddleocr_processor import PaddleOCRProcessor


def main():
    img_path = 'sample_data/20250701OCR测试.png'
    processor = PaddleOCRProcessor()
    results = processor.recognize_text_from_file(img_path)
    print("OCR结果:")
    for i, r in enumerate(results):
        print(f"  {i+1}. {r.text} (置信度: {r.confidence:.2f})")


if __name__ == "__main__":
    main() 