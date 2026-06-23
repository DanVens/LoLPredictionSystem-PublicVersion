from pathlib import Path
import subprocess
import sys

import cv2

CHANNEL = "riotgames"
QUALITY = "best"


def get_streamlink_cmd() -> str:
    local_cmd = Path(sys.executable).with_name("streamlink")
    if local_cmd.exists():
        return str(local_cmd)
    return "streamlink"


def get_hls_url(
    channel: str = CHANNEL,
    quality: str = QUALITY,
    source_url: str | None = None,
) -> str:
    stream_source = source_url or f"twitch.tv/{channel}"
    return subprocess.check_output(
        [get_streamlink_cmd(), stream_source, quality, "--stream-url"],
        text=True,
    ).strip()


def grab_stream_frame(
    channel: str = CHANNEL,
    quality: str = QUALITY,
    source_url: str | None = None,
):
    url = get_hls_url(channel=channel, quality=quality, source_url=source_url)
    cap = cv2.VideoCapture(url)

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise SystemExit("Failed to read frame from stream")
    return frame


if __name__ == "__main__":
    frame = grab_stream_frame()

    cv2.imwrite("frame.png", frame)
    print("Saved frame.png with shape:", frame.shape)
