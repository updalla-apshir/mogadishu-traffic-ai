from ultralytics import YOLO

from anpr.ocr import PlateOCR
from anpr.pipeline import VideoPipeline
from config import Config


def main() -> None:
    config = Config()
    pipeline = VideoPipeline(YOLO(config.model_path), PlateOCR(config.ocr_languages), config)
    pipeline.run()


if __name__ == "__main__":
    main()
