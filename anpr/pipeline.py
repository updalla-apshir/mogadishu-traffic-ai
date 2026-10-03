import cv2

from anpr import renderer
from anpr.geometry import suppress_overlapping
from anpr.ocr import PlateOCR
from anpr.speed import SpeedEstimator
from anpr.tracker import TrackManager
from config import Config


class VideoPipeline:
    def __init__(self, model, ocr: PlateOCR, config: Config):
        self._model = model
        self._ocr = ocr
        self._config = config
        self._tracker = TrackManager(config.class_priority, config.class_names)
        self._speed: SpeedEstimator | None = None

    def run(self) -> None:
        cap = cv2.VideoCapture(self._config.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self._config.video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        non_plate_frames = int(fps * self._config.non_plate_display_seconds)
        grace_frames = int(fps * self._config.plate_grace_seconds)
        self._speed = SpeedEstimator(fps, self._config.pixels_per_meter)

        out = cv2.VideoWriter(self._config.output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        frame_count = 0

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break
                frame_count += 1
                self._process_frame(frame, frame_count, width, height, non_plate_frames, grace_frames)
                out.write(frame)
        finally:
            cap.release()
            out.release()

        self._print_summary()

    def _process_frame(self, frame, frame_count, width, height, non_plate_frames, grace_frames) -> None:
        detections = suppress_overlapping(
            self._detect_and_track(frame, frame_count),
            self._config.class_priority,
            self._config.iou_suppress,
        )

        for det in detections:
            self._tracker.count(det["track_id"], det["class_name"])

        self._tracker.associate_plates(detections)
        active_ids = set()

        for det in detections:
            x1, y1, x2, y2 = det["xyxy"]
            class_name, track_id = det["class_name"], det["track_id"]
            active_ids.add(track_id)

            if class_name == "license_plate":
                self._handle_plate(frame, frame_count, width, height, x1, y1, x2, y2, track_id)
            else:
                if self._speed:
                    self._speed.update(track_id, class_name, (x1, y1, x2, y2), frame_count)
                    speed = self._speed.format_speed(track_id)
                else:
                    speed = ""
                if self._tracker.track_age(track_id, frame_count) <= non_plate_frames:
                    renderer.draw_non_plate(frame, (x1, y1, x2, y2), class_name, track_id)
                    renderer.draw_speed(frame, (x1, y1, x2, y2), speed, y_offset=22)
                else:
                    renderer.draw_speed(frame, (x1, y1, x2, y2), speed)

        for track_id, text in self._tracker.stale_plates(frame_count, grace_frames):
            if track_id not in active_ids:
                renderer.draw_stale_plate(frame, track_id, text)

        renderer.draw_counters(frame, self._tracker.counts)

    def _detect_and_track(self, frame, frame_count: int) -> list[dict]:
        cfg = self._config
        results = self._model.track(frame, persist=True, tracker="bytetrack.yaml",
                                    conf=cfg.min_confidence, verbose=False)
        detections = []

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            for box, tid_t, cls_t, conf_t in zip(boxes.xyxy, boxes.id, boxes.cls, boxes.conf):
                x1, y1, x2, y2 = map(int, box.tolist())
                track_id = int(tid_t.item())
                class_id = int(cls_t.item())
                conf = float(conf_t.item())
                class_name = cfg.class_names.get(class_id, str(class_id))
                resolved = self._tracker.observe(track_id, class_name, conf, frame_count)
                detections.append({
                    "xyxy": (x1, y1, x2, y2),
                    "class_id": cfg.class_id_by_name.get(resolved, class_id),
                    "class_name": resolved,
                    "conf": conf,
                    "track_id": track_id,
                })
        return detections

    def _handle_plate(self, frame, frame_count, width, height, x1, y1, x2, y2, track_id) -> None:
        if frame_count % self._config.ocr_interval == 0:
            crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
            for corrected, confidence, source, raw in self._ocr.read(crop):
                self._tracker.add_ocr(track_id, corrected, confidence)
                print(f"F{frame_count} T{track_id} {source} RAW={raw} OK={corrected} C={confidence:.2f}")
        renderer.draw_plate(frame, (x1, y1, x2, y2), self._tracker.best_plate_text(track_id))

    def _print_summary(self) -> None:
        counts = self._tracker.counts
        print()
        print("=" * 50)
        print("COUNTS")
        print(f"  Vehicles : {counts.get('vehicle', 0)}")
        print(f"  Bajaaj   : {counts.get('bajaaj', 0)}")
        print(f"  Persons  : {counts.get('person', 0)}")
        print("=" * 50)

        records = self._tracker.vehicle_records()
        speeds = self._speed.all_speeds() if self._speed else {}
        if records:
            print()
            print("VEHICLES")
            print(f"{'ID':<6}{'Type':<10}{'Plate':<14}{'Speed':<12}{'Conf':<8}")
            print("-" * 50)
            for rec in records:
                tid = rec["track_id"]
                plate = rec["plate"] or "-"
                speed = f"{speeds[tid]:.0f} km/h" if tid in speeds else "-"
                conf = self._tracker.best_plate_confidence(tid) if rec["plate"] else 0
                conf_str = f"{conf:.0%}" if conf else "-"
                print(f"{tid:<6}{rec['class_name']:<10}{plate:<14}{speed:<12}{conf_str:<8}")
            print("=" * 50)

        print(f"\nSaved: {self._config.output_path}")
