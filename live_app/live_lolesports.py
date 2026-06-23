import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Callable, Optional

import requests
from dotenv import load_dotenv

HL = "en-US"
BASE = "https://esports-api.lolesports.com/persisted/gw"
LIVE_STATS_BASE = "https://feed.lolesports.com/livestats/v1"
MAX_ACCEPTED_FRAME_AGE_S = 600
WINDOW_SEARCH_LOOKBACK_S = 900
WINDOW_SEARCH_STEP_S = 30

EXPORT_ROOT = Path(__file__).resolve().parents[1]
ENV_SEARCH_PATHS = (
    EXPORT_ROOT / ".env",
    EXPORT_ROOT.parent / ".env",
)


def load_lolesports_env() -> Path | None:
    for candidate in ENV_SEARCH_PATHS:
        if candidate.is_file():
            load_dotenv(candidate)
            return candidate
    return None


LOADED_ENV_PATH = load_lolesports_env()
API_KEY = os.getenv("LOLESPORTS_API_KEY")
HEADERS = {"x-api-key": API_KEY} if API_KEY else {}
FEED_HEADERS = {
    "Accept": "application/json",
    "Referer": "https://lolesports.com/",
    "User-Agent": "Mozilla/5.0",
}

ROLE_ALIASES = {
    "top": "top",
    "jungle": "jgl",
    "mid": "mid",
    "middle": "mid",
    "bottom": "bot",
    "bot": "bot",
    "adc": "bot",
    "support": "spt",
    "sup": "spt",
}

STREAM_PROVIDER_PRIORITY = {
    "youtube": 0,
    "afreecatv": 1,
    "soop": 1,
    "twitch": 2,
}


def api_is_configured() -> bool:
    return bool(API_KEY and API_KEY != "YOUR_REAL_KEY_HERE")


def api_disabled_message() -> str:
    searched = ", ".join(str(path) for path in ENV_SEARCH_PATHS)
    return (
        "LoLEsports live data unavailable. Set LOLESPORTS_API_KEY in the environment "
        f"or in one of these files: {searched}."
    )


def require_api_key():
    if not api_is_configured():
        raise RuntimeError(api_disabled_message())


def get_json(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    require_api_key()
    response = requests.get(url, headers=headers or HEADERS, params=params, timeout=20)
    response.raise_for_status()
    try:
        return response.json()
    except JSONDecodeError as exc:
        body = (response.text or "").strip()
        if not body:
            raise RuntimeError(
                "LoLEsports API returned an empty response. This often happens during draft or between games."
            ) from exc
        snippet = body[:200].replace("\n", " ")
        raise RuntimeError(f"LoLEsports API returned non-JSON data: {snippet}") from exc


def find_live_event() -> Optional[dict[str, Any]]:
    live_events = list_live_events()
    if live_events:
        return live_events[0]

    data = get_json(f"{BASE}/getLive", params={"hl": HL})
    events = (((data.get("data") or {}).get("schedule") or {}).get("events")) or []
    for event in events:
        match_id = ((event.get("match") or {}).get("id"))
        if match_id:
            return event

    return None


def list_live_events() -> list[dict[str, Any]]:
    if not api_is_configured():
        return []
    data = get_json(f"{BASE}/getLive", params={"hl": HL})
    events = (((data.get("data") or {}).get("schedule") or {}).get("events")) or []
    live_events: list[dict[str, Any]] = []

    for event in events:
        event_type = (event.get("type") or "").lower()
        state = (event.get("state") or "").lower()
        match_id = ((event.get("match") or {}).get("id"))
        if event_type == "match" and state in {"inprogress", "live"} and match_id:
            live_events.append(event)

    return live_events


def get_event_details(match_id: str) -> dict[str, Any]:
    return get_json(f"{BASE}/getEventDetails", params={"hl": HL, "id": match_id})


def build_stream_url(provider: str | None, parameter: str | None) -> str | None:
    provider_key = (provider or "").strip().lower()
    value = (parameter or "").strip()
    if not value:
        return None
    if value.startswith(("http://", "https://")):
        return value
    if provider_key == "youtube":
        return f"https://www.youtube.com/watch?v={value}"
    if provider_key == "twitch":
        return f"https://www.twitch.tv/{value}"
    if provider_key in {"afreecatv", "soop"}:
        return f"https://play.sooplive.co.kr/{value}"
    return None


def build_stream_candidates(event: dict[str, Any]) -> list[dict[str, Any]]:
    streams = event.get("streams") or []
    candidates: list[dict[str, Any]] = []
    if not isinstance(streams, list):
        return candidates

    for stream in streams:
        if not isinstance(stream, dict):
            continue
        provider = stream.get("provider")
        parameter = stream.get("parameter")
        url = build_stream_url(provider, parameter)
        candidate = {
            "provider": provider,
            "parameter": parameter,
            "url": url,
            "locale": stream.get("locale"),
            "offset": stream.get("offset"),
            "stats_status": stream.get("statsStatus"),
            "countries": stream.get("countries") or [],
        }
        candidates.append(candidate)

    return sorted(
        candidates,
        key=lambda item: (
            STREAM_PROVIDER_PRIORITY.get(str(item.get("provider") or "").lower(), 99),
            0 if item.get("url") else 1,
            str(item.get("locale") or ""),
        ),
    )


def choose_timer_stream(streams: list[dict[str, Any]]) -> dict[str, Any] | None:
    for stream in streams:
        if stream.get("url"):
            return stream
    return streams[0] if streams else None


def extract_team_names(*sources: dict[str, Any]) -> tuple[str | None, str | None]:
    seen_lists: list[list[Any]] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        candidates = [
            deep_get(source, ("match", "teams")),
            deep_get(source, ("event", "match", "teams")),
            deep_get(source, ("event", "teams")),
            deep_get(source, ("teams",)),
        ]
        for teams in candidates:
            if isinstance(teams, list) and teams not in seen_lists:
                seen_lists.append(teams)

    for teams in seen_lists:
        names: list[str] = []
        for team in teams:
            if not isinstance(team, dict):
                continue
            name = (
                deep_get(team, ("name",))
                or deep_get(team, ("code",))
                or deep_get(team, ("team", "name"))
                or deep_get(team, ("team", "code"))
            )
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
        if len(names) >= 2:
            return names[0], names[1]

    return None, None


def list_match_games(event_details: dict[str, Any]) -> list[dict[str, Any]]:
    data = event_details.get("data") or {}
    event = data.get("event") or {}
    match = event.get("match") or event
    games = match.get("games") or []
    if not isinstance(games, list):
        return []

    out = []
    for game in games:
        if "id" not in game:
            continue
        out.append(
            {
                "id": int(game["id"]),
                "number": game.get("number"),
                "state": (game.get("state") or "").lower(),
                "teams": game.get("teams") or [],
            }
        )
    return out


def get_window(game_id: int, starting_time: str | None = None) -> dict[str, Any]:
    params = {"startingTime": starting_time} if starting_time else None
    try:
        return get_json(f"{LIVE_STATS_BASE}/window/{game_id}", params=params, headers=FEED_HEADERS)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Live stats window unavailable for game {game_id}. This often means draft, a transition, or a just-ended game."
        ) from exc


def deep_get(obj: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = obj
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
        if value is None:
            return None
    return value


def to_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def first_int(obj: dict[str, Any], paths: tuple[tuple[str, ...], ...]) -> int | None:
    for path in paths:
        value = to_int(deep_get(obj, path))
        if value is not None:
            return value
    return None


def sum_player_ints(players: list[dict[str, Any]], paths: tuple[tuple[str, ...], ...]) -> int | None:
    total = 0
    found = False
    for player in players:
        value = first_int(player, paths)
        if value is None:
            continue
        total += value
        found = True
    return total if found else None


def extract_team_value(
    team: dict[str, Any],
    team_paths: tuple[tuple[str, ...], ...],
    player_paths: tuple[tuple[str, ...], ...],
) -> int | None:
    direct = first_int(team, team_paths)
    if direct is not None:
        return direct

    players = team.get("players") or team.get("participants") or []
    if not isinstance(players, list):
        return None
    return sum_player_ints(players, player_paths)


def parse_rfc3339(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def format_starting_time(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def align_to_window_step(ts: datetime) -> datetime:
    aligned_second = ts.second - (ts.second % WINDOW_SEARCH_STEP_S)
    return ts.replace(second=aligned_second, microsecond=0)


def local_time_minus_3h(now: datetime | None = None) -> datetime:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc)


def bootstrap_starting_time_candidates(now: datetime | None = None) -> list[str]:
    current = local_time_minus_3h(now)
    base = align_to_window_step(current)
    candidates: list[str] = []
    for seconds_back in range(0, WINDOW_SEARCH_LOOKBACK_S + WINDOW_SEARCH_STEP_S, WINDOW_SEARCH_STEP_S):
        candidates.append(format_starting_time(base - timedelta(seconds=seconds_back)))
    return candidates


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def delay_starting_time_candidates(feed_delay_s: int | None) -> list[str]:
    if feed_delay_s is None:
        return []

    base = align_to_window_step(datetime.now(timezone.utc) - timedelta(seconds=max(0, feed_delay_s)))
    offsets = [0, -30, 30, -60, 60, -90, 90, -120, 120, -150, 150, -180]
    return dedupe_preserve_order(
        [format_starting_time(base + timedelta(seconds=offset)) for offset in offsets]
    )

def incremental_starting_time_candidates(
    last_timestamp: str | None,
    feed_delay_s: int | None = None,
) -> list[str]:
    candidates = delay_starting_time_candidates(feed_delay_s)
    candidates.extend(bootstrap_starting_time_candidates())

    if last_timestamp:
        parsed = parse_rfc3339(last_timestamp)
        if parsed is not None:
            base = align_to_window_step(parsed) + timedelta(seconds=WINDOW_SEARCH_STEP_S)
            for seconds_back in range(0, 181, WINDOW_SEARCH_STEP_S):
                candidates.append(format_starting_time(base - timedelta(seconds=seconds_back)))

    if not candidates:
        candidates.extend(bootstrap_starting_time_candidates())

    return dedupe_preserve_order(candidates)


def learn_feed_delay_s(frame_ts: datetime | None) -> int | None:
    if frame_ts is None:
        return None
    delay = int((datetime.now(timezone.utc) - frame_ts.astimezone(timezone.utc)).total_seconds())
    return max(0, delay)


def frame_age_s(frame: dict[str, Any]) -> int | None:
    return learn_feed_delay_s(frame_timestamp(frame))


def frame_is_fresh(frame: dict[str, Any]) -> bool:
    age = frame_age_s(frame)
    return age is not None and age <= MAX_ACCEPTED_FRAME_AGE_S


def frame_game_state(frame: dict[str, Any]) -> str | None:
    raw_state = (frame or {}).get("gameState")
    if isinstance(raw_state, str):
        return raw_state.strip().lower()
    if isinstance(raw_state, dict):
        nested_state = raw_state.get("state") or raw_state.get("gameState")
        if isinstance(nested_state, str):
            return nested_state.strip().lower()
    return None


def window_game_state(window: dict[str, Any]) -> str | None:
    frames = window.get("frames") or []
    if not frames:
        return None
    return frame_game_state(frames[-1])


def is_terminal_game_state(state: str | None) -> bool:
    return state in {
        "completed",
        "complete",
        "done",
        "ended",
        "end",
        "finished",
        "game_over",
        "post_game",
        "postgame",
    }


def frame_timestamp(frame: dict[str, Any]) -> datetime | None:
    return parse_rfc3339((frame or {}).get("rfc460Timestamp") or "")


def extract_frame_teams(frame: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    game_state = frame.get("gameState")
    if isinstance(game_state, dict):
        return game_state.get("blueTeam") or {}, game_state.get("redTeam") or {}
    return frame.get("blueTeam") or {}, frame.get("redTeam") or {}


def estimate_time_s(frame: dict[str, Any], game_start_time: datetime | None) -> int | None:
    direct = first_int(
        frame,
        (
            ("gameTime",),
            ("gameTimeSeconds",),
            ("gameTimeInSeconds",),
            ("gameClock",),
        ),
    )
    if direct is not None:
        return direct

    game_state = frame.get("gameState")
    if isinstance(game_state, dict):
        direct = first_int(
            game_state,
            (
                ("gameTime",),
                ("gameTimeSeconds",),
                ("gameTimeInSeconds",),
                ("gameClock",),
            ),
        )
        if direct is not None:
            return direct

    frame_ts = frame_timestamp(frame)
    if frame_ts is not None and game_start_time is not None:
        return max(0, int((frame_ts - game_start_time).total_seconds()))

    return None


def build_participant_metadata_map(window: dict[str, Any]) -> dict[int, dict[str, Any]]:
    metadata_map: dict[int, dict[str, Any]] = {}
    game_metadata = window.get("gameMetadata") or {}
    team_metadata_blocks = [
        (game_metadata.get("blueTeamMetadata") or {}).get("participantMetadata") or [],
        (game_metadata.get("redTeamMetadata") or {}).get("participantMetadata") or [],
    ]
    for participants in team_metadata_blocks:
        if not isinstance(participants, list):
            continue
        for participant in participants:
            if not isinstance(participant, dict):
                continue
            participant_id = to_int(participant.get("participantId"))
            if participant_id is None:
                continue
            metadata_map[participant_id] = {
                "participant_id": participant_id,
                "champion_id": participant.get("championId"),
                "summoner_name": participant.get("summonerName"),
                "role": participant.get("role"),
            }
    return metadata_map


def extract_participant_stats(
    team: dict[str, Any],
    participant_metadata_map: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    participants = team.get("participants") or team.get("players") or []
    if not isinstance(participants, list):
        return []

    extracted: list[dict[str, Any]] = []
    for participant in participants:
        if not isinstance(participant, dict):
            continue
        participant_id = to_int(participant.get("participantId"))
        metadata = participant_metadata_map.get(participant_id or -1, {})
        extracted.append(
            {
                "participant_id": participant_id,
                "champion_id": metadata.get("champion_id"),
                "summoner_name": metadata.get("summoner_name"),
                "role": metadata.get("role"),
                "total_gold": to_int(participant.get("totalGold")),
                "level": to_int(participant.get("level")),
                "kills": to_int(participant.get("kills")),
                "deaths": to_int(participant.get("deaths")),
                "assists": to_int(participant.get("assists")),
                "creep_score": to_int(participant.get("creepScore")),
            }
        )
    return extracted


def normalize_role_name(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    normalized = ROLE_ALIASES.get(raw.strip().lower())
    return normalized


def build_role_snapshot(
    participants: list[dict[str, Any]],
    side_prefix: str,
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for participant in participants:
        role = normalize_role_name(participant.get("role"))
        if not role:
            continue
        snapshot[f"{side_prefix}_champion_{role}"] = participant.get("champion_id")
        snapshot[f"{side_prefix}_gold_{role}"] = participant.get("total_gold")
        snapshot[f"{side_prefix}_summoner_{role}"] = participant.get("summoner_name")
    return snapshot


def extract_frame_scoreboard(
    frame: dict[str, Any],
    game_start_time: datetime | None = None,
    participant_metadata_map: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    blue_team, red_team = extract_frame_teams(frame)
    participant_metadata_map = participant_metadata_map or {}
    participants_left = extract_participant_stats(blue_team, participant_metadata_map)
    participants_right = extract_participant_stats(red_team, participant_metadata_map)

    result = {
        "source": "api",
        "gold_left": to_int(blue_team.get("totalGold")),
        "gold_right": to_int(red_team.get("totalGold")),
        "gold_left_raw": None,
        "gold_right_raw": None,
        "kills_left": to_int(blue_team.get("totalKills")),
        "kills_right": to_int(red_team.get("totalKills")),
        "kills_left_raw": None,
        "kills_right_raw": None,
        "time_s": estimate_time_s(frame, game_start_time),
        "time_raw": None,
        "feed_timestamp": (frame.get("rfc460Timestamp") or "").strip() or None,
        "feed_age_s": frame_age_s(frame),
        "participants_left": participants_left,
        "participants_right": participants_right,
        "roi_dir": None,
    }
    result.update(build_role_snapshot(participants_left, "blue"))
    result.update(build_role_snapshot(participants_right, "red"))
    return result


def frame_has_live_stats(frame: dict[str, Any]) -> bool:
    blue_team, red_team = extract_frame_teams(frame)
    stat_candidates = [
        to_int((blue_team or {}).get("totalGold")) or 0,
        to_int((red_team or {}).get("totalGold")) or 0,
        to_int((blue_team or {}).get("totalKills")) or 0,
        to_int((red_team or {}).get("totalKills")) or 0,
        to_int((blue_team or {}).get("towers")) or 0,
        to_int((red_team or {}).get("towers")) or 0,
    ]
    if any(value > 0 for value in stat_candidates):
        return True

    players = (blue_team.get("participants") or []) + (red_team.get("participants") or [])
    levels = [to_int(player.get("level")) or 0 for player in players if isinstance(player, dict)]
    return max(levels, default=0) > 1


def is_bad_starting_time_error(exc: Exception) -> bool:
    return isinstance(exc, requests.HTTPError) and getattr(exc.response, "status_code", None) == 400


def bootstrap_window(
    game_id: int,
    debug: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    best_window = None
    best_score = None
    last_error = None
    attempted_starting_times: list[str] = []

    for starting_time in bootstrap_starting_time_candidates():
        attempted_starting_times.append(starting_time)
        if debug:
            debug(f"bootstrap gameId={game_id} trying startingTime={starting_time}")
        try:
            window = get_window(game_id, starting_time=starting_time)
        except Exception as exc:
            if is_bad_starting_time_error(exc):
                if debug:
                    debug(
                        f"bootstrap gameId={game_id} rejected startingTime={starting_time} with HTTP 400"
                    )
                continue
            last_error = exc
            if debug:
                debug(
                    f"bootstrap gameId={game_id} failed startingTime={starting_time}: {type(exc).__name__}: {exc}"
                )
            continue

        frames = window.get("frames") or []
        if not frames:
            if debug:
                debug(f"bootstrap gameId={game_id} got 0 frames for startingTime={starting_time}")
            continue

        last = frames[-1]
        age = frame_age_s(last)
        if not frame_is_fresh(last):
            if debug:
                debug(
                    "bootstrap gameId="
                    f"{game_id} stale startingTime={starting_time} "
                    f"ts={last.get('rfc460Timestamp')} "
                    f"age={age}s maxAge={MAX_ACCEPTED_FRAME_AGE_S}s"
                )
            continue
        blue, red = extract_frame_teams(last)
        ts = frame_timestamp(last)
        score = (
            1 if frame_has_live_stats(last) else 0,
            int(ts.timestamp()) if ts else 0,
            to_int(blue.get("totalGold")) or 0,
            to_int(red.get("totalGold")) or 0,
        )
        if best_score is None or score > best_score:
            best_window = window
            best_score = score
            if debug:
                debug(
                    "bootstrap gameId="
                    f"{game_id} accepted startingTime={starting_time} "
                    f"ts={last.get('rfc460Timestamp')} "
                    f"age={age}s "
                    f"blueGold={to_int(blue.get('totalGold'))} "
                    f"redGold={to_int(red.get('totalGold'))} "
                    f"blueKills={to_int(blue.get('totalKills'))} "
                    f"redKills={to_int(red.get('totalKills'))}"
                )
            if score[0] > 0:
                return window

    if best_window is not None:
        return best_window

    if last_error is not None:
        raise RuntimeError(
            "Live stats window unavailable for "
            f"game {game_id}. Tried startingTime values: {', '.join(attempted_starting_times)}. "
            f"Last error: {type(last_error).__name__}: {last_error}"
        ) from last_error
    raise RuntimeError(
        "Could not find a usable live stats window for "
        f"game {game_id}. Tried startingTime values: {', '.join(attempted_starting_times)}."
    )


def choose_live_game(games: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not games:
        return None

    prioritized_games = sorted(
        games,
        key=lambda game: (
            0 if game.get("state") in {"inprogress", "live"} else 1,
            -(to_int(game.get("number")) or 0),
        ),
    )
    return prioritized_games[0] if prioritized_games else None


def get_best_window(
    game_id: int,
    last_timestamp: str | None = None,
    feed_delay_s: int | None = None,
    debug: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    best_window = None
    best_score = None
    attempted_starting_times: list[str] = []
    last_error: Exception | None = None

    for starting_time in incremental_starting_time_candidates(last_timestamp, feed_delay_s=feed_delay_s):
        attempted_starting_times.append(starting_time)
        if debug:
            debug(f"window gameId={game_id} trying startingTime={starting_time}")
        try:
            window = get_window(game_id, starting_time=starting_time)
        except Exception as exc:
            if is_bad_starting_time_error(exc):
                if debug:
                    debug(f"window gameId={game_id} rejected startingTime={starting_time} with HTTP 400")
                continue
            last_error = exc
            if debug:
                debug(
                    f"window gameId={game_id} failed startingTime={starting_time}: {type(exc).__name__}: {exc}"
                )
            continue
        if not (window.get("frames") or []):
            if debug:
                debug(f"window gameId={game_id} got 0 frames for startingTime={starting_time}")
            continue
        frame = pick_best_frame(window.get("frames") or [])
        age = frame_age_s(frame)
        if not frame_is_fresh(frame):
            if debug:
                debug(
                    "window gameId="
                    f"{game_id} stale startingTime={starting_time} "
                    f"ts={frame.get('rfc460Timestamp')} "
                    f"age={age}s maxAge={MAX_ACCEPTED_FRAME_AGE_S}s"
                )
            continue
        score = score_window(window)
        if best_score is None or score > best_score:
            best_window = window
            best_score = score
            blue, red = extract_frame_teams(frame)
            if debug:
                debug(
                    "window gameId="
                    f"{game_id} accepted startingTime={starting_time} "
                    f"ts={frame.get('rfc460Timestamp')} "
                    f"age={age}s "
                    f"blueGold={to_int(blue.get('totalGold'))} "
                    f"redGold={to_int(red.get('totalGold'))} "
                    f"blueKills={to_int(blue.get('totalKills'))} "
                    f"redKills={to_int(red.get('totalKills'))}"
                )
            if score[0] > 0:
                return window

    if best_window is not None and best_score and best_score[0] > 0:
        return best_window

    try:
        fallback = bootstrap_window(game_id, debug=debug)
    except Exception as fallback_error:
        combined_attempts = attempted_starting_times.copy()
        if not combined_attempts:
            combined_attempts = incremental_starting_time_candidates(
                last_timestamp,
                feed_delay_s=feed_delay_s,
            )
        raise RuntimeError(
            "Live stats window unavailable for "
            f"game {game_id}. Tried startingTime values: {', '.join(combined_attempts)}. "
            f"Last error: {type(fallback_error).__name__}: {fallback_error}"
        ) from fallback_error
    fallback_score = score_window(fallback)
    if best_window is None or fallback_score > (best_score or (-1, -1, -1, -1)):
        return fallback
    return best_window


def score_frame(frame: dict[str, Any]) -> tuple[int, int, int, int]:
    blue_team, red_team = extract_frame_teams(frame)
    ts = frame_timestamp(frame)
    return (
        1 if frame_has_live_stats(frame) else 0,
        int(ts.timestamp()) if ts else 0,
        to_int(blue_team.get("totalGold")) or 0,
        to_int(red_team.get("totalGold")) or 0,
    )


def pick_best_frame(frames: list[dict[str, Any]]) -> dict[str, Any]:
    if not frames:
        raise RuntimeError("Live stats window has no frames.")
    return max(frames, key=score_frame)


def score_window(window: dict[str, Any]) -> tuple[int, int, int, int]:
    frames = window.get("frames") or []
    if not frames:
        return (0, 0, 0, 0)
    return score_frame(pick_best_frame(frames))


def merge_monotonic_metric(current: int | None, previous: int | None) -> int | None:
    if previous is None:
        return current
    if current is None or current < previous:
        return previous
    return current


@dataclass
class LiveGameContext:
    league: str | None
    match_id: str
    game_id: int
    game_number: int | None
    team_left: str | None
    team_right: str | None
    streams: list[dict[str, Any]]
    timer_stream: dict[str, Any] | None


@dataclass
class LiveMatchSnapshot:
    league: str | None
    match_id: str
    games: list[dict[str, Any]]
    team_left: str | None
    team_right: str | None
    team_lookup: dict[str, str]
    streams: list[dict[str, Any]]
    timer_stream: dict[str, Any] | None


def build_team_lookup(*sources: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    seen_lists: list[list[Any]] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        candidates = [
            deep_get(source, ("match", "teams")),
            deep_get(source, ("event", "match", "teams")),
            deep_get(source, ("event", "teams")),
            deep_get(source, ("teams",)),
        ]
        for teams in candidates:
            if isinstance(teams, list) and teams not in seen_lists:
                seen_lists.append(teams)

    for teams in seen_lists:
        for team in teams:
            if not isinstance(team, dict):
                continue
            team_id = team.get("id")
            if team_id is None:
                continue
            name = (
                deep_get(team, ("name",))
                or deep_get(team, ("code",))
                or deep_get(team, ("team", "name"))
                or deep_get(team, ("team", "code"))
            )
            if isinstance(name, str) and name.strip():
                lookup[str(team_id)] = name.strip()
    return lookup


def resolve_game_side_teams(
    team_lookup: dict[str, str],
    game: dict[str, Any] | None,
) -> tuple[str | None, str | None]:
    if not isinstance(game, dict):
        return None, None
    blue_name = None
    red_name = None
    for team_ref in game.get("teams") or []:
        if not isinstance(team_ref, dict):
            continue
        side = (team_ref.get("side") or "").strip().lower()
        team_id = team_ref.get("id")
        name = team_lookup.get(str(team_id)) if team_id is not None else None
        if side == "blue" and name:
            blue_name = name
        elif side == "red" and name:
            red_name = name
    return blue_name, red_name


def build_live_match_snapshot(live_event: dict[str, Any]) -> LiveMatchSnapshot | None:
    if not isinstance(live_event, dict):
        return None

    match = live_event.get("match") or {}
    match_id = match.get("id")
    if not match_id:
        return None

    details = get_event_details(str(match_id))
    event = (details.get("data") or {}).get("event") or {}
    games = list_match_games(details)
    league = (live_event.get("league") or {}).get("name")
    team_left, team_right = extract_team_names(live_event, details.get("data") or {}, details)
    team_lookup = build_team_lookup(live_event, details.get("data") or {}, details)
    streams = build_stream_candidates(event)
    return LiveMatchSnapshot(
        league=league,
        match_id=str(match_id),
        games=games,
        team_left=team_left,
        team_right=team_right,
        team_lookup=team_lookup,
        streams=streams,
        timer_stream=choose_timer_stream(streams),
    )


def list_live_match_snapshots() -> list[LiveMatchSnapshot]:
    if not api_is_configured():
        return []
    snapshots: list[LiveMatchSnapshot] = []
    for live_event in list_live_events():
        snapshot = build_live_match_snapshot(live_event)
        if snapshot is not None:
            snapshots.append(snapshot)
    return snapshots


class LolesportsLiveClient:
    def __init__(self, debug: bool = False):
        self.context: LiveGameContext | None = None
        self.last_timestamp: str | None = None
        self.feed_delay_s: int | None = None
        self.game_start_time: datetime | None = None
        self.last_result: dict[str, Any] | None = None
        self.last_result_at: float | None = None
        self.debug = debug or (os.getenv("LOLESPORTS_DEBUG") == "1")

    def _debug(self, message: str):
        if self.debug:
            print(f"[lolesports] {message}", file=sys.stderr, flush=True)

    def current_match_snapshot(self) -> LiveMatchSnapshot | None:
        if not api_is_configured():
            return None
        live_event = find_live_event()
        if not live_event:
            return None
        return build_live_match_snapshot(live_event)

    def current_live_game_id(self) -> int | None:
        snapshot = self.current_match_snapshot()
        if snapshot is None:
            return None
        game = choose_live_game(snapshot.games)
        if not game or game.get("state") not in {"inprogress", "live"}:
            return None
        return game["id"]

    def build_status_result(
        self,
        status: str,
        message: str,
        snapshot: LiveMatchSnapshot | None = None,
        game: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = {
            "source": "api",
            "status": status,
            "status_message": message,
            "gold_left": None,
            "gold_right": None,
            "gold_left_raw": None,
            "gold_right_raw": None,
            "kills_left": None,
            "kills_right": None,
            "kills_left_raw": None,
            "kills_right_raw": None,
            "time_s": None,
            "time_raw": None,
            "feed_timestamp": None,
            "feed_age_s": None,
            "patch_version": None,
            "participants_left": [],
            "participants_right": [],
            "roi_dir": None,
        }
        for side_prefix in ("blue", "red"):
            for role in ("top", "jgl", "mid", "bot", "spt"):
                result[f"{side_prefix}_champion_{role}"] = None
                result[f"{side_prefix}_gold_{role}"] = None
                result[f"{side_prefix}_summoner_{role}"] = None
        if snapshot is not None:
            result["league"] = snapshot.league
            result["match_id"] = snapshot.match_id
            result["stream_candidates"] = snapshot.streams
            result["timer_stream"] = snapshot.timer_stream
            blue_name, red_name = resolve_game_side_teams(snapshot.team_lookup, game)
            result["team_left"] = blue_name or snapshot.team_left
            result["team_right"] = red_name or snapshot.team_right
        elif self.context is not None:
            result["league"] = self.context.league
            result["match_id"] = self.context.match_id
            result["team_left"] = self.context.team_left
            result["team_right"] = self.context.team_right
            result["stream_candidates"] = self.context.streams
            result["timer_stream"] = self.context.timer_stream

        if game is not None:
            result["game_id"] = game.get("id")
            result["game_number"] = to_int(game.get("number"))
            result["game_state"] = game.get("state")
        elif self.context is not None:
            result["game_id"] = self.context.game_id
            result["game_number"] = self.context.game_number

        return result

    def build_window_ended_result(
        self,
        snapshot: LiveMatchSnapshot | None,
        game: dict[str, Any] | None,
        payload_state: str | None,
        reset_context: bool = True,
    ) -> dict[str, Any]:
        previous_context = self.context
        if reset_context:
            self.reset()
        result = self.build_status_result(
            "between_games",
            "Current game ended. Waiting for the next game to start.",
            snapshot=snapshot,
            game=game,
        )
        if previous_context is not None and result.get("game_id") is None:
            result["league"] = previous_context.league
            result["match_id"] = previous_context.match_id
            result["game_id"] = previous_context.game_id
            result["game_number"] = previous_context.game_number
            result["team_left"] = previous_context.team_left
            result["team_right"] = previous_context.team_right
            result["stream_candidates"] = previous_context.streams
            result["timer_stream"] = previous_context.timer_stream
        result["game_state"] = payload_state or result.get("game_state")
        result["payload_game_state"] = payload_state
        return result

    def window_ended_result(
        self,
        window: dict[str, Any],
        snapshot: LiveMatchSnapshot | None,
        game: dict[str, Any] | None,
        reset_context: bool = True,
    ) -> dict[str, Any] | None:
        state = window_game_state(window)
        if not is_terminal_game_state(state):
            return None
        self._debug(
            f"window payload indicates game ended gameId={self.context.game_id if self.context else (game or {}).get('id')} "
            f"payloadGameState={state}"
        )
        return self.build_window_ended_result(snapshot, game, state, reset_context=reset_context)

    def current_match_status(
        self,
        snapshot: LiveMatchSnapshot | None,
    ) -> tuple[str, str, dict[str, Any] | None]:
        if snapshot is None:
            return ("no_live_event", "No live LoLEsports event found.", None)

        live_game = next(
            (game for game in snapshot.games if game.get("state") in {"inprogress", "live"}),
            None,
        )
        if live_game is not None:
            return ("in_progress", "Game is in progress.", live_game)

        if any(game.get("state") == "completed" for game in snapshot.games) and any(
            game.get("state") in {"unstarted", "draft"} for game in snapshot.games
        ):
            completed = max(
                (game for game in snapshot.games if game.get("state") == "completed"),
                key=lambda game: to_int(game.get("number")) or 0,
            )
            return (
                "between_games",
                "Current game ended. Waiting for the next game to start.",
                completed,
            )

        if snapshot.games and all(game.get("state") == "completed" for game in snapshot.games):
            finished = max(snapshot.games, key=lambda game: to_int(game.get("number")) or 0)
            return ("match_ended", "Match ended.", finished)

        next_game = min(
            (game for game in snapshot.games if game.get("state") == "unstarted"),
            key=lambda game: to_int(game.get("number")) or 0,
            default=None,
        )
        if next_game is not None:
            return ("pre_game", "Waiting for the next game to start.", next_game)

        return ("unknown", "Live match found, but no in-progress game is available yet.", None)

    def summarize_snapshot(
        self,
        snapshot: LiveMatchSnapshot,
        featured: bool = False,
    ) -> dict[str, Any]:
        status, message, game = self.current_match_status(snapshot)
        summary = {
            "league": snapshot.league,
            "match_id": snapshot.match_id,
            "status": status,
            "status_message": message,
            "featured": featured,
            "stream_candidates": snapshot.streams,
            "timer_stream": snapshot.timer_stream,
        }
        if game is not None:
            summary["game_id"] = game.get("id")
            summary["game_number"] = to_int(game.get("number"))
            summary["game_state"] = game.get("state")
        blue_name, red_name = resolve_game_side_teams(snapshot.team_lookup, game)
        summary["team_left"] = blue_name or snapshot.team_left
        summary["team_right"] = red_name or snapshot.team_right
        return summary

    def result_for_snapshot(
        self,
        snapshot: LiveMatchSnapshot,
        featured: bool = False,
    ) -> dict[str, Any]:
        status, message, game = self.current_match_status(snapshot)
        if status != "in_progress" or game is None:
            result = self.build_status_result(status, message, snapshot=snapshot, game=game)
            result["featured"] = featured
            return result

        context = LiveGameContext(
            league=snapshot.league,
            match_id=snapshot.match_id,
            game_id=int(game["id"]),
            game_number=to_int(game.get("number")),
            team_left=resolve_game_side_teams(snapshot.team_lookup, game)[0] or snapshot.team_left,
            team_right=resolve_game_side_teams(snapshot.team_lookup, game)[1] or snapshot.team_right,
            streams=snapshot.streams,
            timer_stream=snapshot.timer_stream,
        )

        try:
            window = get_best_window(
                context.game_id,
                last_timestamp=self.last_timestamp if featured and self.context and self.context.game_id == context.game_id else None,
                feed_delay_s=self.feed_delay_s,
                debug=self._debug if featured else None,
            )
        except Exception as exc:
            result = self.build_status_result("window_unavailable", str(exc), snapshot=snapshot, game=game)
            result["featured"] = featured
            return result

        ended_result = self.window_ended_result(window, snapshot, game, reset_context=featured)
        if ended_result is not None:
            ended_result["featured"] = featured
            return ended_result

        frames = window.get("frames") or []
        if not frames:
            result = self.build_status_result(
                "window_unavailable",
                "Live stats window returned no frames.",
                snapshot=snapshot,
                game=game,
            )
            result["featured"] = featured
            return result

        game_start_time = frame_timestamp(frames[0])
        frame = pick_best_frame(frames)
        result = extract_frame_scoreboard(
            frame,
            game_start_time,
            participant_metadata_map=build_participant_metadata_map(window),
        )
        result["league"] = context.league
        result["match_id"] = context.match_id
        result["game_id"] = context.game_id
        result["game_number"] = context.game_number
        result["team_left"] = context.team_left
        result["team_right"] = context.team_right
        result["stream_candidates"] = snapshot.streams
        result["timer_stream"] = snapshot.timer_stream
        result["game_state"] = game.get("state")
        result["patch_version"] = deep_get(window, ("gameMetadata", "patchVersion"))
        result["time_source"] = "api_estimate" if not featured else result.get("time_source")
        result["status"] = "in_progress"
        result["status_message"] = "Game is in progress."
        result["featured"] = featured
        return result

    def list_live_games(
        self,
        featured_snapshot: LiveMatchSnapshot | None = None,
        featured_result: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        seen_match_ids: set[str] = set()

        snapshots = list_live_match_snapshots()
        if featured_snapshot is not None:
            snapshots = [
                featured_snapshot,
                *[snapshot for snapshot in snapshots if snapshot.match_id != featured_snapshot.match_id],
            ]

        for snapshot in snapshots:
            if snapshot.match_id in seen_match_ids:
                continue
            if (
                featured_result is not None
                and featured_snapshot is not None
                and snapshot.match_id == featured_snapshot.match_id
            ):
                result = dict(featured_result)
                result["featured"] = True
            else:
                result = self.result_for_snapshot(snapshot, featured=False)
            summaries.append(result)
            seen_match_ids.add(snapshot.match_id)

        return summaries

    def reset(self):
        self.context = None
        self.last_timestamp = None
        self.game_start_time = None
        self.last_result = None
        self.last_result_at = None

    def build_context_for_game(
        self,
        snapshot: LiveMatchSnapshot,
        game: dict[str, Any],
    ) -> LiveGameContext:
        blue_name, red_name = resolve_game_side_teams(snapshot.team_lookup, game)
        return LiveGameContext(
            league=snapshot.league,
            match_id=snapshot.match_id,
            game_id=int(game["id"]),
            game_number=to_int(game.get("number")),
            team_left=blue_name or snapshot.team_left,
            team_right=red_name or snapshot.team_right,
            streams=snapshot.streams,
            timer_stream=snapshot.timer_stream,
        )

    def try_fetch_featured_snapshot(
        self,
        snapshot: LiveMatchSnapshot,
    ) -> tuple[dict[str, Any] | None, Exception | None]:
        status, _, game = self.current_match_status(snapshot)
        if status != "in_progress" or game is None:
            return None, None

        game_id = int(game["id"])
        same_context = (
            self.context is not None
            and self.context.match_id == snapshot.match_id
            and self.context.game_id == game_id
        )
        if not same_context:
            self.context = self.build_context_for_game(snapshot, game)
            self.last_timestamp = None
            self.game_start_time = None
            self.last_result = None
            self.last_result_at = None

        try:
            window = get_best_window(
                game_id,
                last_timestamp=self.last_timestamp if same_context else None,
                feed_delay_s=self.feed_delay_s,
                debug=self._debug,
            )
        except Exception as exc:
            if not same_context:
                self.reset()
            return None, exc

        ended_result = self.window_ended_result(window, snapshot, game)
        if ended_result is not None:
            return ended_result, None

        return self.ingest_window(window), None

    def discover_live_game(
        self,
        snapshot: LiveMatchSnapshot | None = None,
    ) -> tuple[LiveGameContext, dict[str, Any]]:
        snapshot = snapshot or self.current_match_snapshot()
        if snapshot is None:
            raise RuntimeError("No live LoLEsports event found.")

        games = snapshot.games
        if not games:
            raise RuntimeError(f"Live match {snapshot.match_id} has no game ids yet.")

        self._debug(
            f"live matchId={snapshot.match_id} league={snapshot.league or 'Unknown League'} games={json.dumps(games)}"
        )
        best_game = choose_live_game(games)
        if best_game is None or best_game.get("state") not in {"inprogress", "live"}:
            raise RuntimeError("Could not choose a live LoLEsports game id.")

        self._debug(
            f"selected gameId={best_game['id']} gameNumber={best_game.get('number')} state={best_game.get('state')}"
        )
        best_window = get_best_window(
            best_game["id"],
            feed_delay_s=self.feed_delay_s,
            debug=self._debug,
        )

        context = self.build_context_for_game(snapshot, best_game)
        self.context = context
        self.last_timestamp = None
        self.game_start_time = None
        self.last_result = None
        return context, best_window

    def ingest_window(self, window: dict[str, Any]) -> dict[str, Any]:
        frames = window.get("frames") or []
        if not frames:
            if self.last_result is not None:
                return self.last_result
            raise RuntimeError("Live stats window has no frames.")

        if self.game_start_time is None:
            self.game_start_time = frame_timestamp(frames[0])

        frame = pick_best_frame(frames)
        frame_ts = frame_timestamp(frame)
        last_ts = (frame.get("rfc460Timestamp") or "").strip() or None
        if last_ts:
            self.last_timestamp = last_ts
        learned_delay = learn_feed_delay_s(frame_ts)
        if learned_delay is not None:
            self.feed_delay_s = learned_delay
            self._debug(
                f"learned feed delay gameId={self.context.game_id if self.context else 'unknown'} "
                f"delay={self.feed_delay_s}s frameTs={last_ts}"
            )

        result = extract_frame_scoreboard(
            frame,
            self.game_start_time,
            participant_metadata_map=build_participant_metadata_map(window),
        )
        if self.context:
            result["league"] = self.context.league
            result["match_id"] = self.context.match_id
            result["game_id"] = self.context.game_id
            result["game_number"] = self.context.game_number
            result["team_left"] = self.context.team_left
            result["team_right"] = self.context.team_right
            result["stream_candidates"] = self.context.streams
            result["timer_stream"] = self.context.timer_stream
            result["status"] = "in_progress"
            result["status_message"] = "Game is in progress."
        result["patch_version"] = deep_get(window, ("gameMetadata", "patchVersion"))

        if self.last_result is not None:
            result["gold_left"] = merge_monotonic_metric(
                result.get("gold_left"), self.last_result.get("gold_left")
            )
            result["gold_right"] = merge_monotonic_metric(
                result.get("gold_right"), self.last_result.get("gold_right")
            )
            result["kills_left"] = merge_monotonic_metric(
                result.get("kills_left"), self.last_result.get("kills_left")
            )
            result["kills_right"] = merge_monotonic_metric(
                result.get("kills_right"), self.last_result.get("kills_right")
            )
            result["time_s"] = merge_monotonic_metric(
                result.get("time_s"), self.last_result.get("time_s")
            )
            for side_prefix in ("blue", "red"):
                for role in ("top", "jgl", "mid", "bot", "spt"):
                    gold_key = f"{side_prefix}_gold_{role}"
                    result[gold_key] = merge_monotonic_metric(
                        result.get(gold_key), self.last_result.get(gold_key)
                    )
                    for info_key in (
                        f"{side_prefix}_champion_{role}",
                        f"{side_prefix}_summoner_{role}",
                    ):
                        if result.get(info_key) is None:
                            result[info_key] = self.last_result.get(info_key)

            if result.get("time_s") is not None and self.last_result_at is not None:
                elapsed = max(0, int(time.time() - self.last_result_at))
                prior_time = self.last_result.get("time_s")
                if prior_time is not None and result["time_s"] == prior_time:
                    # Keep the clock moving when the feed repeats or falls back to an old frame.
                    result["time_s"] = prior_time + elapsed

        if frame_ts is not None and self.game_start_time is None:
            self.game_start_time = frame_ts

        self.last_result = result
        self.last_result_at = time.time()
        return result

    def fetch_scoreboard(self) -> dict[str, Any]:
        if not api_is_configured():
            if self.context is not None:
                self.reset()
            result = self.build_status_result("api_unavailable", api_disabled_message())
            result["live_games"] = []
            result["time_source"] = "api_unavailable"
            return result

        snapshots = list_live_match_snapshots()
        if not snapshots:
            if self.context is not None:
                self.reset()
            result = self.build_status_result("no_live_event", "No live LoLEsports event found.")
            result["live_games"] = []
            return result

        ordered_snapshots = snapshots
        if self.context is not None:
            ordered_snapshots = [
                *[snapshot for snapshot in snapshots if snapshot.match_id == self.context.match_id],
                *[snapshot for snapshot in snapshots if snapshot.match_id != self.context.match_id],
            ]

        fallback_result: dict[str, Any] | None = None
        last_error: Exception | None = None

        for snapshot in ordered_snapshots:
            result, error = self.try_fetch_featured_snapshot(snapshot)
            if error is not None:
                last_error = error
                self._debug(
                    f"featured candidate failed matchId={snapshot.match_id}: {type(error).__name__}: {error}"
                )
                if fallback_result is None:
                    status, message, game = self.current_match_status(snapshot)
                    fallback_result = self.build_status_result(
                        "window_unavailable" if status == "in_progress" else status,
                        str(error) if status == "in_progress" else message,
                        snapshot=snapshot,
                        game=game,
                    )
                continue
            if result is None:
                if fallback_result is None:
                    status, message, game = self.current_match_status(snapshot)
                    fallback_result = self.build_status_result(
                        status,
                        message,
                        snapshot=snapshot,
                        game=game,
                    )
                continue

            result["featured"] = True
            result["live_games"] = self.list_live_games(
                featured_snapshot=snapshot,
                featured_result=result,
            )
            return result

        if self.context is not None:
            self.reset()

        result = fallback_result or self.build_status_result(
            "window_unavailable",
            f"No usable live stats window found. Last error: {last_error}" if last_error else "No usable live stats window found.",
        )
        result["live_games"] = self.list_live_games()
        return result


_DEFAULT_CLIENT = LolesportsLiveClient()


def fetch_live_scoreboard(client: LolesportsLiveClient | None = None) -> dict[str, Any]:
    active_client = client or _DEFAULT_CLIENT
    return active_client.fetch_scoreboard()


def main():
    client = LolesportsLiveClient(debug=True)
    print("Finding live match…")
    scoreboard = fetch_live_scoreboard(client)
    print(
        f"Live: {scoreboard.get('league') or 'Unknown League'} | "
        f"matchId={scoreboard['match_id']} | gameId={scoreboard['game_id']}"
    )
    print("Polling API every 5s… (Ctrl+C to stop)\n")

    while True:
        print(json.dumps(fetch_live_scoreboard(client), sort_keys=True))
        time.sleep(5)


if __name__ == "__main__":
    main()
