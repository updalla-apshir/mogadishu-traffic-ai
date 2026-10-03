def box_iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def suppress_overlapping(detections: list[dict], class_priority: dict, iou_threshold: float) -> list[dict]:
    ordered = sorted(detections, key=lambda d: (-class_priority.get(d["class_name"], 0), -d["conf"]))
    kept = []
    for det in ordered:
        if not any(box_iou(det["xyxy"], k["xyxy"]) > iou_threshold for k in kept):
            kept.append(det)
    return kept
