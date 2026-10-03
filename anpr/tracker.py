from collections import defaultdict

from anpr.ocr import vote_best, is_valid_plate


class TrackManager:
    def __init__(self, class_priority: dict, class_names: dict):
        self._priority = class_priority
        self._start_frame: dict[int, int] = {}
        self._last_seen: dict[int, int] = {}
        self._class_votes: dict[int, dict[str, float]] = {}
        self._ocr_history: dict[int, list[tuple[str, float]]] = defaultdict(list)
        self._counts: dict[str, set[int]] = defaultdict(set)
        self._plate_links: dict[int, str] = {}
        self._plate_conf: dict[int, float] = {}

    def observe(self, track_id: int, class_name: str, conf: float, frame: int) -> str:
        if track_id not in self._start_frame:
            self._start_frame[track_id] = frame
        self._last_seen[track_id] = frame

        votes = self._class_votes.setdefault(track_id, {})
        votes[class_name] = votes.get(class_name, 0.0) + conf

        resolved = max(votes, key=lambda c: votes[c])
        if len(votes) > 1:
            top = sorted(votes.items(), key=lambda kv: -kv[1])
            if top[0][1] - top[1][1] < 0.3:
                resolved = max([top[0][0], top[1][0]], key=lambda c: self._priority.get(c, 0))
        return resolved

    def track_age(self, track_id: int, frame: int) -> int:
        return frame - self._start_frame.get(track_id, frame)

    def count(self, track_id: int, class_name: str) -> None:
        if class_name in ("vehicle", "bajaaj", "person"):
            self._counts[class_name].add(track_id)

    @property
    def counts(self) -> dict[str, int]:
        return {name: len(ids) for name, ids in self._counts.items()}

    def add_ocr(self, track_id: int, text: str, confidence: float) -> None:
        self._ocr_history[track_id].append((text, confidence))

    def best_plate_text(self, track_id: int) -> str:
        return vote_best(self._ocr_history.get(track_id, []))

    def best_plate_confidence(self, track_id: int) -> float:
        best = self.best_plate_text(track_id)
        if not best:
            return 0.0
        return max((c for t, c in self._ocr_history.get(track_id, []) if t == best), default=0.0)

    def stale_plates(self, frame: int, grace: int) -> list[tuple[int, str]]:
        result = []
        for tid, last in self._last_seen.items():
            if "license_plate" in self._class_votes.get(tid, {}) and frame - last <= grace:
                text = self.best_plate_text(tid)
                if text:
                    result.append((tid, text))
        return result

    def associate_plates(self, detections: list[dict]) -> dict[int, str]:
        vehicles = [d for d in detections if d["class_name"] in ("vehicle", "bajaaj")]
        plates = [d for d in detections if d["class_name"] == "license_plate"]

        for plate in plates:
            px1, py1, px2, py2 = plate["xyxy"]
            pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
            text = self.best_plate_text(plate["track_id"])
            if not text:
                continue
            conf = self.best_plate_confidence(plate["track_id"])

            best_veh, best_dist = None, float("inf")
            for veh in vehicles:
                vx1, vy1, vx2, vy2 = veh["xyxy"]
                if vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2:
                    best_veh, best_dist = veh["track_id"], 0.0
                    break
                vcx, vcy = (vx1 + vx2) / 2, (vy1 + vy2) / 2
                dist = ((pcx - vcx) ** 2 + (pcy - vcy) ** 2) ** 0.5
                if dist < best_dist:
                    best_veh, best_dist = veh["track_id"], dist

            if best_veh is not None and best_dist < 200:
                prev_conf = self._plate_conf.get(best_veh, 0.0)
                if best_veh not in self._plate_links:
                    self._plate_links[best_veh] = text
                    self._plate_conf[best_veh] = conf
                elif text != self._plate_links[best_veh] and conf > prev_conf + 0.1:
                    self._plate_links[best_veh] = text
                    self._plate_conf[best_veh] = conf
                elif text == self._plate_links[best_veh] and conf > prev_conf:
                    self._plate_conf[best_veh] = conf

        return dict(self._plate_links)

    def vehicle_records(self) -> list[dict]:
        records = []
        for tid, votes in self._class_votes.items():
            name = max(votes, key=lambda c: votes[c])
            if len(votes) > 1:
                top = sorted(votes.items(), key=lambda kv: -kv[1])
                if top[0][1] - top[1][1] < 0.3:
                    name = max([top[0][0], top[1][0]], key=lambda c: self._priority.get(c, 0))
            if name in ("vehicle", "bajaaj"):
                records.append({
                    "track_id": tid,
                    "class_name": name,
                    "plate": self._plate_links.get(tid, ""),
                    "start_frame": self._start_frame.get(tid, 0),
                })
        records.sort(key=lambda r: r["start_frame"])
        return records
