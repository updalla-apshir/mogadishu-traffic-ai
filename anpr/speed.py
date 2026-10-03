import math
from collections import defaultdict, deque

SPEED_CLASSES = ("vehicle", "bajaaj")
_HISTORY = 30
_MIN_SAMPLES = 5
_SMOOTH = 0.3


class SpeedEstimator:
    def __init__(self, fps: float, pixels_per_meter: float):
        self._fps = fps if fps > 0 else 30.0
        self._ppm = max(pixels_per_meter, 1e-6)
        self._history: dict[int, deque] = defaultdict(lambda: deque(maxlen=_HISTORY))
        self._smoothed: dict[int, float] = {}

    def update(self, track_id: int, class_name: str, xyxy: tuple, frame: int) -> None:
        if class_name not in SPEED_CLASSES:
            return
        x1, y1, x2, y2 = xyxy
        self._history[track_id].append((frame, (x1 + x2) / 2, (y1 + y2) / 2))
        speed = self._compute(track_id)
        if speed is None:
            return
        prev = self._smoothed.get(track_id)
        self._smoothed[track_id] = speed if prev is None else _SMOOTH * speed + (1 - _SMOOTH) * prev

    def _compute(self, track_id: int) -> float | None:
        hist = self._history[track_id]
        if len(hist) < _MIN_SAMPLES:
            return None
        f0, x0, y0 = hist[0]
        f1, x1, y1 = hist[-1]
        frames = f1 - f0
        if frames <= 0:
            return None
        dist_m = math.hypot(x1 - x0, y1 - y0) / self._ppm
        seconds = frames / self._fps
        return (dist_m / seconds) * 3.6 if seconds > 0 else None

    def format_speed(self, track_id: int) -> str:
        s = self._smoothed.get(track_id)
        return f"{s:.0f} km/h" if s is not None else ""

    def all_speeds(self) -> dict[int, float]:
        return dict(self._smoothed)
