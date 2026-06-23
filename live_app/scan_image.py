import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Callable

import cv2
import numpy as np
import pytesseract


@dataclass(frozen=True)
class FieldSpec:
    whitelist: str
    psm: int
    crop_fracs: tuple[float, ...]
    parser: Callable[[str], int | None]
    exact_pattern: re.Pattern[str]
    replacements: dict[str, str]


def load_profile(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def scale_box(box, w, h, base_w, base_h):
    x1, y1, x2, y2 = box
    sx = w / base_w
    sy = h / base_h
    return (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))


def crop(frame, box):
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return frame[0:1, 0:1]
    return frame[y1:y2, x1:x2]


def preprocess_variants(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return (
        ("gray", gray),
        ("gray_inv", 255 - gray),
        ("otsu", otsu),
        ("otsu_inv", 255 - otsu),
    )


def parse_gold(s: str) -> int | None:
    t = s.upper().replace(",", ".").replace(" ", "")
    m = re.search(r"(\d{1,3}(?:\.\d)?)K", t)
    if not m:
        return None

    value = int(float(m.group(1)) * 1000)
    return value if 1_000 <= value <= 100_000 else None

def parse_time_s(s: str) -> int | None:
    m = re.search(r"(\d{1,2}):(\d{2})", s)
    if not m:
        return None
    mm, ss = int(m.group(1)), int(m.group(2))
    value = mm * 60 + ss
    return value if ss < 60 and value <= 5_400 else None

def parse_int(s: str) -> int | None:
    m = re.search(r"\d+", s)
    if not m:
        return None
    value = int(m.group(0))
    return value if value <= 100 else None


def normalize_text(text: str, spec: FieldSpec) -> str:
    cleaned = []
    for ch in text.strip().upper():
        ch = spec.replacements.get(ch, ch)
        if ch:
            cleaned.append(ch)
    return "".join(cleaned)


def candidate_score(text: str, parsed: int | None, spec: FieldSpec) -> int:
    if not text:
        return -10_000

    score = 0
    if parsed is not None:
        score += 100

    match = spec.exact_pattern.search(text)
    if match:
        if match.group(0) == text:
            score += 40
        else:
            score += 20
            score -= len(text) - len(match.group(0))

    if any(ch.isdigit() for ch in text):
        score += 8
    score -= max(0, len(text) - 6)
    return score


def ocr_field(frame, box, spec: FieldSpec) -> str:
    roi = crop(frame, box)
    if roi.size == 0:
        return ""

    candidates = []
    value_votes = Counter()

    for crop_frac in spec.crop_fracs:
        cut_h = max(1, int(roi.shape[0] * crop_frac))
        roi_slice = roi[:cut_h, :]

        for _, variant in preprocess_variants(roi_slice):
            cfg = f"--oem 3 --psm {spec.psm} -c tessedit_char_whitelist={spec.whitelist}"
            raw = pytesseract.image_to_string(variant, config=cfg).strip()
            text = normalize_text(raw, spec)
            parsed = spec.parser(text)
            score = candidate_score(text, parsed, spec)
            candidates.append((score, text, parsed))

            if parsed is not None:
                value_votes[parsed] += 1

    if not candidates:
        return ""

    best_score, best_text, _ = max(
        candidates,
        key=lambda item: (
            item[0] + value_votes.get(item[2], 0) * 8,
            -len(item[1]),
            item[1],
        ),
    )

    if best_score < 0:
        return ""
    return best_text


def gold_char_variants(char_img):
    blur = cv2.GaussianBlur(char_img, (3, 3), 0)
    otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(char_img)
    sharp = cv2.addWeighted(
        char_img,
        1.8,
        cv2.GaussianBlur(char_img, (0, 0), 1.2),
        -0.8,
        0,
    )

    return (
        char_img,
        255 - char_img,
        blur,
        255 - blur,
        otsu,
        255 - otsu,
        clahe,
        255 - clahe,
        sharp,
        255 - sharp,
    )


def ocr_single_gold_char(char_img, whitelist: str) -> str:
    votes = Counter()
    allowed = set(whitelist)
    replacements = {"k": "K", "l": "1", "I": "1", "|": "1"}

    for psm in (8, 13):
        for variant in gold_char_variants(char_img):
            padded = cv2.copyMakeBorder(
                variant,
                20,
                20,
                20,
                20,
                cv2.BORDER_CONSTANT,
                value=0,
            )
            raw = pytesseract.image_to_string(
                padded,
                config=f"--oem 3 --psm {psm} -c tessedit_char_whitelist={whitelist}",
            ).strip()
            if not raw:
                continue

            char = replacements.get(raw, raw).upper()
            if len(char) == 1 and char in allowed:
                votes[char] += 1

    if not votes:
        return ""
    return max(votes.items(), key=lambda item: (item[1], item[0]))[0]


def ocr_gold_components(frame, box) -> str:
    roi = crop(frame, box)
    if roi.size == 0:
        return ""

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    big = cv2.resize(gray, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    mask = (big >= 180).astype("uint8")

    comp_count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    comps = []
    for i in range(1, comp_count):
        x, y, w, h, area = stats[i]
        if area < 300 or h < 80:
            continue
        comps.append((x, y, w, h))

    if not comps:
        return ""

    top = min(y for _, y, _, _ in comps)
    comps = [comp for comp in comps if comp[1] <= top + 15]
    comps.sort(key=lambda comp: comp[0])

    chars = []
    for x, y, w, h in comps:
        char_img = big[
            max(0, y - 10):min(big.shape[0], y + h + 10),
            max(0, x - 10):min(big.shape[1], x + w + 10),
        ]
        whitelist = "Kk0123456789"
        chars.append(ocr_single_gold_char(char_img, whitelist))

    if not chars or "K" not in chars:
        return ""

    k_index = chars.index("K")
    digits = [char for char in chars[:k_index] if char.isdigit()]
    if len(digits) < 3:
        return ""

    text = f"{digits[-3]}{digits[-2]}.{digits[-1]}K"
    return text if parse_gold(text) is not None else ""


def scale_rois(profile: dict, frame_shape) -> dict[str, tuple[int, int, int, int]]:
    h, w = frame_shape[:2]
    rois = profile["rois"]
    base_w = profile["base_width"]
    base_h = profile["base_height"]
    return {
        "gold_left": scale_box(rois["gold_left"], w, h, base_w, base_h),
        "gold_right": scale_box(rois["gold_right"], w, h, base_w, base_h),
        "time": scale_box(rois["time"], w, h, base_w, base_h),
        "kills_left": scale_box(rois["kills_left"], w, h, base_w, base_h),
        "kills_right": scale_box(rois["kills_right"], w, h, base_w, base_h),
    }


def save_roi_crops(img, boxes: dict[str, tuple[int, int, int, int]], outdir: str):
    import os

    os.makedirs(outdir, exist_ok=True)
    cv2.imwrite(f"{outdir}/gold_left.png", crop(img, boxes["gold_left"]))
    cv2.imwrite(f"{outdir}/gold_right.png", crop(img, boxes["gold_right"]))
    cv2.imwrite(f"{outdir}/time.png", crop(img, boxes["time"]))
    cv2.imwrite(f"{outdir}/kills_left.png", crop(img, boxes["kills_left"]))
    cv2.imwrite(f"{outdir}/kills_right.png", crop(img, boxes["kills_right"]))


def scan_time_only(img, profile: dict, outdir: str | None = None) -> dict:
    boxes = scale_rois(profile, img.shape)

    if outdir:
        import os

        os.makedirs(outdir, exist_ok=True)
        cv2.imwrite(f"{outdir}/time.png", crop(img, boxes["time"]))

    time_raw = ocr_field(img, boxes["time"], TIME_SPEC)
    return {
        "time_raw": time_raw,
        "time_s": parse_time_s(time_raw),
        "roi_dir": outdir,
    }


def scan_frame(img, profile: dict, outdir: str | None = None) -> dict:
    boxes = scale_rois(profile, img.shape)

    if outdir:
        save_roi_crops(img, boxes, outdir)

    gold_l_raw = ocr_gold_components(img, boxes["gold_left"]) or ocr_field(img, boxes["gold_left"], GOLD_SPEC)
    gold_r_raw = ocr_gold_components(img, boxes["gold_right"]) or ocr_field(img, boxes["gold_right"], GOLD_SPEC)
    time_raw = ocr_field(img, boxes["time"], TIME_SPEC)
    kill_l_raw = ocr_field(img, boxes["kills_left"], INT_SPEC)
    kill_r_raw = ocr_field(img, boxes["kills_right"], INT_SPEC)

    return {
        "gold_left_raw": gold_l_raw,
        "gold_right_raw": gold_r_raw,
        "time_raw": time_raw,
        "kills_left_raw": kill_l_raw,
        "kills_right_raw": kill_r_raw,
        "gold_left": parse_gold(gold_l_raw),
        "gold_right": parse_gold(gold_r_raw),
        "time_s": parse_time_s(time_raw),
        "kills_left": parse_int(kill_l_raw),
        "kills_right": parse_int(kill_r_raw),
        "roi_dir": outdir,
    }


GOLD_SPEC = FieldSpec(
    whitelist="0123456789.kK",
    psm=7,
    crop_fracs=(1.0, 0.92, 0.85, 0.75),
    parser=parse_gold,
    exact_pattern=re.compile(r"^\d{1,3}(?:\.\d)?K$"),
    replacements={
        " ": "",
        ",": ".",
        ":": ".",
        ";": ".",
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "|": "1",
        "!": "1",
        "S": "5",
        "B": "8",
    },
)

TIME_SPEC = FieldSpec(
    whitelist="0123456789:",
    psm=7,
    crop_fracs=(1.0, 0.92),
    parser=parse_time_s,
    exact_pattern=re.compile(r"^\d{1,2}:\d{2}$"),
    replacements={
        " ": "",
        ".": ":",
        ",": ":",
        ";": ":",
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "|": "1",
        "!": "1",
        "S": "5",
        "B": "8",
    },
)

INT_SPEC = FieldSpec(
    whitelist="0123456789",
    psm=7,
    crop_fracs=(1.0, 0.92),
    parser=parse_int,
    exact_pattern=re.compile(r"^\d{1,3}$"),
    replacements={
        " ": "",
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "|": "1",
        "!": "1",
        "S": "5",
        "B": "8",
    },
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image_path")
    p.add_argument("--profile", default=DEFAULT_PROFILE_PATH)
    p.add_argument("--outdir", default="debug_rois")
    args = p.parse_args()

    profile = load_profile(args.profile)

    img = cv2.imread(args.image_path)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image_path}")

    out = scan_frame(img, profile, outdir=args.outdir)

    print(out)
    print(f"Saved ROI crops in: {args.outdir}/")

if __name__ == "__main__":
    main()
EXPORT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_PATH = str(EXPORT_ROOT / "profiles/lcs_1152p.json")
