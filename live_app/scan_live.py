import argparse
import json
from pathlib import Path
import time

import cv2

from grab_frame import grab_stream_frame
from scan_image import load_profile, scan_frame, scan_time_only


EXPORT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_PATH = str(EXPORT_ROOT / "profiles/lcs_1152p.json")


def scan_stream_once(
    profile: dict,
    channel: str = "riotgames",
    quality: str = "best",
    source_url: str | None = None,
    save_frame: str | None = None,
    roi_dir: str | None = None,
):
    frame = grab_stream_frame(channel=channel, quality=quality, source_url=source_url)

    if save_frame:
        cv2.imwrite(save_frame, frame)

    result = scan_frame(frame, profile, outdir=roi_dir)
    result["source"] = "ocr"
    return result


def scan_stream_time_once(
    profile: dict,
    channel: str = "riotgames",
    quality: str = "best",
    source_url: str | None = None,
    save_frame: str | None = None,
    roi_dir: str | None = None,
):
    frame = grab_stream_frame(channel=channel, quality=quality, source_url=source_url)

    if save_frame:
        cv2.imwrite(save_frame, frame)

    result = scan_time_only(frame, profile, outdir=roi_dir)
    result["source"] = "ocr_time"
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--channel", default="riotgames")
    p.add_argument("--url", default=None, help="Optional direct stream URL for streamlink.")
    p.add_argument("--quality", default="best")
    p.add_argument("--profile", default=DEFAULT_PROFILE_PATH)
    p.add_argument("--interval", type=float, default=5.0)
    p.add_argument("--once", action="store_true")
    p.add_argument("--save-frame", default=None, help="Optional path to save the grabbed frame")
    p.add_argument(
        "--roi-dir",
        default=None,
        help="Optional directory to save ROI crops from the most recent frame",
    )
    args = p.parse_args()

    profile = load_profile(args.profile)

    while True:
        result = scan_stream_once(
            profile,
            channel=args.channel,
            quality=args.quality,
            source_url=args.url,
            save_frame=args.save_frame,
            roi_dir=args.roi_dir,
        )
        print(json.dumps(result, sort_keys=True))

        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
