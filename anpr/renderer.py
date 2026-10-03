import cv2

GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
ORANGE = (0, 200, 255)
WHITE = (255, 255, 255)


def draw_plate(frame, xyxy: tuple, text: str = "") -> None:
    x1, y1, x2, y2 = xyxy
    cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
    if text:
        cv2.putText(frame, text, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)


def draw_non_plate(frame, xyxy: tuple, class_name: str, track_id: int) -> None:
    x1, y1, x2, y2 = xyxy
    cv2.rectangle(frame, (x1, y1), (x2, y2), YELLOW, 2)
    cv2.putText(frame, f"{class_name} ID:{track_id}", (x1, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)


def draw_speed(frame, xyxy: tuple, speed_text: str, y_offset: int = 0) -> None:
    if not speed_text:
        return
    x1, y1, x2, y2 = xyxy
    cv2.putText(frame, speed_text, (x1, max(22, y1 - 8) + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def draw_stale_plate(frame, track_id: int, text: str) -> None:
    cv2.putText(frame, text, (20, 130 + 20 * (track_id % 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, ORANGE, 2)


def draw_counters(frame, counts: dict[str, int]) -> None:
    for label, y in [
        (f"Vehicles: {counts.get('vehicle', 0)}", 30),
        (f"Bajaaj: {counts.get('bajaaj', 0)}", 60),
        (f"Persons: {counts.get('person', 0)}", 90),
    ]:
        cv2.putText(frame, label, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
