import re

import cv2
import easyocr


def normalize_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def correct_plate(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return ""

    repl = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "G": "6", "B": "8"}

    if len(text) >= 2 and text[0] == "A":
        prefix = text[:2]
        if prefix[1] in "0OQG":
            prefix = "AD"
        elif prefix[1] == "E":
            prefix = "AE"
        if prefix in ("AD", "AE"):
            return prefix + "".join(repl.get(c, c) for c in text[2:])

    return "".join(repl.get(c, c) for c in text)


def is_valid_plate(text: str) -> bool:
    if not text or len(text) < 3:
        return False
    if not any(c.isdigit() for c in text):
        return False
    if len(set(text)) == 1:
        return False
    if not any(c.isalpha() for c in text) and len(text) > 6:
        return False
    return True


def vote_best(history: list[tuple[str, float]]) -> str:
    valid = [(t, c) for t, c in history if is_valid_plate(t)]
    if not valid:
        return ""

    stats: dict[str, tuple[float, int]] = {}
    for text, conf in valid:
        max_c, count = stats.get(text, (0.0, 0))
        stats[text] = (max(max_c, conf), count + 1)

    def score(item):
        _, (max_conf, count) = item
        return max_conf * (1.0 + 0.06 * min(count, 5))

    return max(stats.items(), key=score)[0]


class PlateOCR:
    def __init__(self, languages: tuple = ("en",)):
        self._reader = easyocr.Reader(list(languages))

    def read(self, crop) -> list[tuple[str, float, str, str]]:
        if crop is None or crop.size == 0:
            return []

        h, w = crop.shape[:2]
        padded = cv2.copyMakeBorder(crop, int(h * 0.2), int(h * 0.2), int(w * 0.15), int(w * 0.15),
                                    cv2.BORDER_CONSTANT, value=(255, 255, 255))
        enlarged = cv2.resize(padded, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        results = []
        for name, img in [("original", enlarged), ("gray", gray), ("threshold", thresh)]:
            for _, text, conf in self._reader.readtext(img, detail=1, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"):
                text = normalize_text(text)
                corrected = correct_plate(text)
                if is_valid_plate(corrected):
                    results.append((corrected, float(conf), name, text))
        return results
