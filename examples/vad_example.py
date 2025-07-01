"""
SileroVADProcessor VAD调用示例
"""
from src.processors.vad.silero_vad_processor import SileroVADProcessor


def main():
    audio_path = 'sample_data/1417796_input.wav_10.wav'
    processor = SileroVADProcessor()
    segments = processor.detect_voice_segments_from_file(audio_path)
    print("VAD检测结果:")
    for i, seg in enumerate(segments):
        print(f"  {i+1}. start: {seg.start:.2f}s, end: {seg.end:.2f}s")


if __name__ == "__main__":
    main() 