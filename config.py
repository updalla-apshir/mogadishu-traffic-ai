from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    model_path: str = r"./models/train/weights/best.pt"
    video_path: str = r"./mogadishu_traffic_25s.mp4"
    output_path: str = r"./output.mp4"

    ocr_interval: int = 5
    ocr_languages: tuple = ("en",)

    non_plate_display_seconds: float = 1.0
    plate_grace_seconds: float = 0.5

    min_confidence: float = 0.5
    iou_suppress: float = 0.45
    pixels_per_meter: float = 45.0

    class_names: dict = field(default_factory=lambda: {
        0: "bajaaj",
        1: "license_plate",
        2: "person",
        3: "vehicle",
    })

    class_priority: dict = field(default_factory=lambda: {
        "license_plate": 4,
        "bajaaj": 3,
        "vehicle": 2,
        "person": 1,
    })

    @property
    def class_id_by_name(self) -> dict:
        return {name: cid for cid, name in self.class_names.items()}
