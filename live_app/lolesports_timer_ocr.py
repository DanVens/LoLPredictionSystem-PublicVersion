from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
from playwright.sync_api import Page, sync_playwright

from scan_image import load_profile, scan_time_only


DEFAULT_URL = "https://lolesports.com/live"
EXPORT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_PATH = str(EXPORT_ROOT / "profiles/lcs_1152p.json")


@dataclass
class TimerOCRResult:
    time_raw: str | None
    time_s: int | None
    source: str
    screenshot_path: str | None = None
    roi_dir: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "time_raw": self.time_raw,
            "time_s": self.time_s,
            "source": self.source,
            "screenshot_path": self.screenshot_path,
            "roi_dir": self.roi_dir,
            "error": self.error,
        }


@dataclass
class LivePageMatchInfo:
    source: str
    page_url: str | None
    league: str | None
    league_slug: str | None
    team_left: str | None
    team_right: str | None
    match_id: str | None
    status: str
    status_message: str
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "page_url": self.page_url,
            "league": self.league,
            "league_slug": self.league_slug,
            "team_left": self.team_left,
            "team_right": self.team_right,
            "match_id": self.match_id,
            "status": self.status,
            "status_message": self.status_message,
            "error": self.error,
        }


def screenshot_bytes_to_frame(raw: bytes):
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Could not decode browser screenshot.")
    return frame


def prepare_lolesports_page(page: Page, url: str, wait_ms: int) -> None:
    page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    page.wait_for_timeout(wait_ms)

    # These selectors are intentionally broad because consent/play overlays vary by locale/provider.
    click_texts = (
        "Accept All",
        "Accept all",
        "Accept",
        "I Agree",
        "Agree",
        "Continue",
        "Play",
    )
    for text in click_texts:
        try:
            page.get_by_text(text, exact=False).first.click(timeout=900)
            page.wait_for_timeout(500)
        except Exception:
            pass

    try:
        page.mouse.click(960, 540)
        page.wait_for_timeout(500)
    except Exception:
        pass


def display_league_name(league_slug: str | None) -> str | None:
    if not league_slug:
        return None
    normalized = league_slug.strip().lower()
    known = {
        "lck": "LCK",
        "lec": "LEC",
        "lcs": "LCS",
        "lpl": "LPL",
        "lcp": "LCP",
        "cblol": "CBLOL",
    }
    return known.get(normalized, normalized.replace("-", " ").upper())


class LolesportsTimerOCR:
    def __init__(
        self,
        profile_path: str = DEFAULT_PROFILE_PATH,
        url: str = DEFAULT_URL,
        viewport_width: int = 1920,
        viewport_height: int = 1080,
        headless: bool = True,
        wait_ms: int = 5000,
    ) -> None:
        self.profile = load_profile(profile_path)
        self.url = url
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.headless = headless
        self.wait_ms = wait_ms

        self._playwright = None
        self._browser = None
        self._page = None
        self._prepared = False

    def start(self) -> None:
        if self._page is not None:
            return
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--autoplay-policy=no-user-gesture-required",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        context = self._browser.new_context(
            viewport={"width": self.viewport_width, "height": self.viewport_height},
            device_scale_factor=1,
            locale="en-US",
        )
        self._page = context.new_page()

    def close(self) -> None:
        if self._browser is not None:
            self._browser.close()
        if self._playwright is not None:
            self._playwright.stop()
        self._playwright = None
        self._browser = None
        self._page = None
        self._prepared = False

    def capture_time(
        self,
        screenshot_path: str | None = None,
        roi_dir: str | None = None,
    ) -> TimerOCRResult:
        try:
            self.start()
            assert self._page is not None
            if not self._prepared:
                prepare_lolesports_page(self._page, self.url, self.wait_ms)
                self._prepared = True

            raw = self._page.screenshot(path=screenshot_path, full_page=False)
            frame = screenshot_bytes_to_frame(raw)
            result = scan_time_only(frame, self.profile, outdir=roi_dir)
            return TimerOCRResult(
                time_raw=result.get("time_raw"),
                time_s=result.get("time_s"),
                source="lolesports_browser_ocr",
                screenshot_path=screenshot_path,
                roi_dir=roi_dir,
            )
        except Exception as exc:
            return TimerOCRResult(
                time_raw=None,
                time_s=None,
                source="lolesports_browser_ocr",
                screenshot_path=screenshot_path,
                roi_dir=roi_dir,
                error=str(exc),
            )

    def capture_match_info(self) -> LivePageMatchInfo:
        try:
            self.start()
            assert self._page is not None
            if not self._prepared:
                prepare_lolesports_page(self._page, self.url, self.wait_ms)
                self._prepared = True

            page_url = self._page.url or self.url
            parsed = urlparse(page_url)
            parts = [part for part in parsed.path.split("/") if part]
            league_slug = parts[1] if len(parts) >= 2 and parts[0] == "live" else None
            league = display_league_name(league_slug)

            payload = self._page.evaluate(
                """
                () => {
                  const tricodes = Array.from(
                    document.querySelectorAll(".event-header .tricode")
                  )
                    .map((node) => (node.textContent || "").trim())
                    .filter(Boolean);
                  const bodyText = (document.body?.innerText || "").replace(/\\s+/g, " ").trim();
                  return {
                    tricodes,
                    statsUnavailable: bodyText.includes("Stats are currently unavailable."),
                  };
                }
                """
            )
            tricodes = payload.get("tricodes") or []
            team_left = tricodes[0] if len(tricodes) >= 1 else None
            team_right = tricodes[1] if len(tricodes) >= 2 else None
            stats_unavailable = bool(payload.get("statsUnavailable"))

            if team_left and team_right:
                status_message = (
                    "Recovered matchup from lolesports.com/live, but live stats are currently unavailable."
                    if stats_unavailable
                    else "Recovered matchup from lolesports.com/live."
                )
                match_id = f"live_page:{league_slug or 'unknown'}:{team_left.lower()}_vs_{team_right.lower()}"
                return LivePageMatchInfo(
                    source="lolesports_live_page",
                    page_url=page_url,
                    league=league,
                    league_slug=league_slug,
                    team_left=team_left,
                    team_right=team_right,
                    match_id=match_id,
                    status="page_match_only",
                    status_message=status_message,
                )

            return LivePageMatchInfo(
                source="lolesports_live_page",
                page_url=page_url,
                league=league,
                league_slug=league_slug,
                team_left=team_left,
                team_right=team_right,
                match_id=None,
                status="no_live_event",
                status_message="Could not find a live matchup on lolesports.com/live.",
                error="matchup selectors returned no teams",
            )
        except Exception as exc:
            return LivePageMatchInfo(
                source="lolesports_live_page",
                page_url=None,
                league=None,
                league_slug=None,
                team_left=None,
                team_right=None,
                match_id=None,
                status="live_page_error",
                status_message="Could not read live matchup from lolesports.com/live.",
                error=str(exc),
            )

    def __enter__(self) -> "LolesportsTimerOCR":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="OCR only the timer from lolesports.com/live.")
    parser.add_argument("--profile", default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--headful", action="store_true", help="Show the browser window for debugging.")
    parser.add_argument("--wait-ms", type=int, default=5000)
    parser.add_argument("--screenshot", default="debug_lolesports_timer.png")
    parser.add_argument("--roi-dir", default="debug_lolesports_timer_roi")
    args = parser.parse_args()

    with LolesportsTimerOCR(
        profile_path=args.profile,
        url=args.url,
        viewport_width=args.width,
        viewport_height=args.height,
        headless=not args.headful,
        wait_ms=args.wait_ms,
    ) as timer:
        result = timer.capture_time(screenshot_path=args.screenshot, roi_dir=args.roi_dir)
    print(result.as_dict())
    return 0 if result.error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
