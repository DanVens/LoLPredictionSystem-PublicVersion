import argparse
from collections import deque
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import threading
import time

from flask import Flask, Response, jsonify, request

from live_inference import LiveSequencePredictor
from live_lolesports import LolesportsLiveClient, api_is_configured, fetch_live_scoreboard
from lolesports_timer_ocr import LolesportsTimerOCR
from scan_image import load_profile
from scan_live import scan_stream_once


ROLE_ORDER = ("top", "jgl", "mid", "bot", "spt")
EXPORT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_PATH = str(EXPORT_ROOT / "profiles/lcs_1152p.json")
DEFAULT_LCK_PROFILE_PATH = str(EXPORT_ROOT / "profiles/lck_1078p.json")
DEFAULT_LIVE_LOG_DIR = str(EXPORT_ROOT / "live_prediction_logs")
DEFAULT_REPLAY_OUTPUT_DIR = str(EXPORT_ROOT / "replay_prediction_logs")
DEFAULT_REPLAY_CSV_PATH = str(EXPORT_ROOT / "data/training_table_all.csv")


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Live LoL Prediction Interface</title>
  <style>
    :root {
      --bg: #0d1321;
      --panel: #1d2d44;
      --panel-2: #22324c;
      --text: #f0ebd8;
      --muted: #b7b19c;
      --accent: #d4a373;
      --good: #7bd389;
      --bad: #ef6f6c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top, rgba(212,163,115,0.18), transparent 28%),
        linear-gradient(160deg, var(--bg), #08101d 70%);
      color: var(--text);
      min-height: 100vh;
    }
    main {
      max-width: 980px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: clamp(2rem, 4vw, 3.5rem);
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .sub {
      color: var(--muted);
      margin-bottom: 28px;
    }
    .matchup {
      margin: 0 0 18px;
      color: var(--accent);
      font-size: 1.1rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(29,45,68,0.8);
      border: 1px solid rgba(240,235,216,0.15);
      margin-bottom: 24px;
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 20px rgba(212,163,115,0.7);
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      margin-bottom: 18px;
    }
    .card {
      background: linear-gradient(180deg, rgba(34,50,76,0.92), rgba(16,25,39,0.92));
      border: 1px solid rgba(240,235,216,0.12);
      border-radius: 22px;
      padding: 22px;
      box-shadow: 0 18px 50px rgba(0,0,0,0.2);
    }
    .label {
      color: var(--muted);
      font-size: 0.82rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      margin-bottom: 10px;
    }
    .value {
      font-size: clamp(2rem, 5vw, 4rem);
      line-height: 1;
      font-weight: 700;
    }
    .meta {
      margin-top: 10px;
      color: var(--muted);
      font-size: 0.95rem;
    }
    .wide {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 18px;
      margin-bottom: 18px;
    }
    .players {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      margin-top: 18px;
    }
    .player-list {
      display: grid;
      gap: 10px;
    }
    .player-row {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(8, 16, 29, 0.72);
      border: 1px solid rgba(240,235,216,0.06);
    }
    .player-main {
      min-width: 0;
    }
    .player-name {
      font-weight: 700;
      color: var(--text);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .player-meta {
      color: var(--muted);
      font-size: 0.88rem;
      margin-top: 2px;
    }
    .player-gold {
      font-weight: 700;
      white-space: nowrap;
    }
    .live-list {
      display: grid;
      gap: 12px;
    }
    .live-item {
      padding: 14px 16px;
      border-radius: 16px;
      background: rgba(8, 16, 29, 0.72);
      border: 1px solid rgba(240,235,216,0.08);
      cursor: pointer;
      transition: border-color 120ms ease, transform 120ms ease, box-shadow 120ms ease;
    }
    .live-item.featured {
      border-color: rgba(212,163,115,0.45);
      box-shadow: 0 0 0 1px rgba(212,163,115,0.18) inset;
    }
    .live-item.selected {
      border-color: rgba(123,211,137,0.6);
      box-shadow: 0 0 0 1px rgba(123,211,137,0.2) inset;
      transform: translateY(-1px);
    }
    .live-item strong {
      display: block;
      margin-bottom: 4px;
      color: var(--text);
    }
    .live-meta {
      color: var(--muted);
      font-size: 0.92rem;
    }
    .live-stats {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    .live-stat {
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(29,45,68,0.55);
    }
    .live-stat-label {
      color: var(--muted);
      font-size: 0.76rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 6px;
    }
    .live-stat-value {
      font-size: 1.15rem;
      font-weight: 700;
    }
    .model-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .model-card {
      padding: 14px 16px;
      border-radius: 16px;
      background: rgba(8, 16, 29, 0.72);
      border: 1px solid rgba(240,235,216,0.08);
    }
    .model-card.primary {
      border-color: rgba(212,163,115,0.45);
      box-shadow: 0 0 0 1px rgba(212,163,115,0.18) inset;
    }
    .model-card-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      margin-bottom: 8px;
    }
    .model-card-title {
      font-weight: 700;
      color: var(--text);
    }
    .model-card-state {
      color: var(--muted);
      font-size: 0.84rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .model-card-values {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-size: 1.2rem;
      font-weight: 700;
    }
    .model-card-meta {
      margin-top: 8px;
      color: var(--muted);
      font-size: 0.9rem;
    }
    .table-wrap {
      overflow: auto;
      border-radius: 16px;
      border: 1px solid rgba(240,235,216,0.08);
      background: rgba(8, 16, 29, 0.72);
    }
    .history-wrap {
      max-height: 280px;
      overflow-x: auto;
      overflow-y: scroll;
      scrollbar-gutter: stable;
    }
    table {
      width: 100%;
      border-collapse: collapse;
    }
    th, td {
      padding: 10px 12px;
      text-align: left;
      border-bottom: 1px solid rgba(240,235,216,0.06);
      white-space: nowrap;
    }
    th {
      color: var(--muted);
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      background: rgba(29,45,68,0.45);
    }
    .manual-grid {
      display: grid;
      gap: 14px;
    }
    .manual-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    .btn {
      border: 1px solid rgba(240,235,216,0.14);
      background: rgba(29,45,68,0.85);
      color: var(--text);
      padding: 10px 14px;
      border-radius: 12px;
      cursor: pointer;
      font: inherit;
    }
    .btn:hover {
      border-color: rgba(212,163,115,0.5);
    }
    textarea {
      width: 100%;
      min-height: 260px;
      padding: 14px;
      border-radius: 16px;
      border: 1px solid rgba(240,235,216,0.08);
      background: rgba(8, 16, 29, 0.92);
      color: var(--text);
      font: 0.92rem/1.45 monospace;
      resize: vertical;
    }
    input[type="text"] {
      width: 100%;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid rgba(240,235,216,0.08);
      background: rgba(8, 16, 29, 0.92);
      color: var(--text);
      font: inherit;
    }
    .inline-grid {
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 10px;
    }
    pre {
      margin: 0;
      padding: 18px;
      border-radius: 18px;
      background: rgba(8, 16, 29, 0.92);
      overflow: auto;
      border: 1px solid rgba(240,235,216,0.08);
      color: var(--text);
      font-size: 0.92rem;
    }
    .error {
      color: var(--bad);
    }
    @media (max-width: 760px) {
      .grid, .wide, .players, .live-stats, .model-grid, .inline-grid { grid-template-columns: 1fr; }
      main { padding: 20px 14px 40px; }
    }
  </style>
</head>
<body>
  <main>
    <h1>Live LoL Prediction Interface</h1>
    <div class="sub">Local live match feed with real-time gold-based win probability estimates</div>
    <div class="matchup" id="matchup">-</div>
    <div class="status">
      <div class="dot"></div>
      <div id="status-text">Starting scanner…</div>
    </div>

    <section class="grid">
      <article class="card">
        <div class="label">Blue Gold</div>
        <div class="value" id="gold-left">-</div>
        <div class="meta" id="gold-left-raw">raw: -</div>
      </article>
      <article class="card">
        <div class="label">Red Gold</div>
        <div class="value" id="gold-right">-</div>
        <div class="meta" id="gold-right-raw">raw: -</div>
      </article>
    </section>

    <section class="wide">
      <article class="card">
        <div class="label">Game Time</div>
        <div class="value" id="time">-</div>
        <div class="meta" id="time-raw">raw: -</div>
      </article>
      <article class="card">
        <div class="label">Gold Lead (Blue)</div>
        <div class="value" id="gold-diff">-</div>
        <div class="meta" id="gold-diff-raw">derived from team gold totals</div>
      </article>
      <article class="card">
        <div class="label">Model Spread</div>
        <div class="value" id="pred-spread">-</div>
        <div class="meta" id="pred-spread-raw">largest gap across model predictions</div>
      </article>
    </section>

    <section class="grid">
      <article class="card">
        <div class="label">Consensus Blue Win Probability</div>
        <div class="value" id="pred-blue">-</div>
        <div class="meta" id="pred-meta">average of available model predictions</div>
      </article>
      <article class="card">
        <div class="label">Consensus Red Win Probability</div>
        <div class="value" id="pred-red">-</div>
        <div class="meta" id="pred-steps">timesteps: -</div>
      </article>
    </section>

    <section class="card" style="margin-top: 18px;">
      <div class="label">Model Odds</div>
      <div class="meta" id="model-grid-meta" style="margin-top: 0; margin-bottom: 12px;">models ready: -</div>
      <div class="model-grid" id="model-grid"></div>
    </section>

    <section class="players">
      <article class="card">
        <div class="label">Blue Champions</div>
        <div class="player-list" id="players-left"></div>
      </article>
      <article class="card">
        <div class="label">Red Champions</div>
        <div class="player-list" id="players-right"></div>
      </article>
    </section>

    <section class="card" style="margin-top: 18px;">
      <div class="label">Live Matches</div>
      <div class="live-list" id="live-games"></div>
    </section>

    <section class="card" style="margin-top: 18px;">
      <div class="label">Saved Prediction History</div>
      <div class="meta" id="history-meta">No saved rows yet.</div>
      <div class="table-wrap history-wrap" style="margin-top: 12px;">
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Consensus</th>
              <th>GRU</th>
              <th>LogReg</th>
              <th>XGBoost</th>
              <th>MLP</th>
            </tr>
          </thead>
          <tbody id="history-body">
            <tr><td colspan="6" class="live-meta">No history yet.</td></tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="card" style="margin-top: 18px;">
      <div class="label">Manual Snapshot</div>
        <div class="manual-grid">
        <div class="meta" id="manual-meta">Paste a snapshot JSON here to override live data and run all models against it.</div>
        <div class="inline-grid">
          <input type="text" id="replay-file-path" value="data/training_table_all.csv" placeholder="Path to snapshot table CSV">
          <input type="text" id="replay-game-id" value="" placeholder="Optional game_id">
        </div>
        <div class="manual-actions">
          <button class="btn" id="manual-template-btn" type="button">Use Current As Template</button>
          <button class="btn" id="manual-apply-btn" type="button">Apply Manual Snapshot</button>
          <button class="btn" id="manual-clear-btn" type="button">Clear Manual Override</button>
          <button class="btn" id="replay-file-btn" type="button">Replay Snapshot Table</button>
        </div>
        <textarea id="manual-json">{
  "source": "manual",
  "status": "in_progress",
  "league": "LEC",
  "source_season": "S16",
  "tournament": "LEC 2026 Summer",
  "tournament_stage": "Playoffs",
  "match_id": "manual_match_1",
  "game_id": 900001,
  "game_number": 1,
  "team_left": "Blue Team",
  "team_right": "Red Team",
  "time_s": 600,
  "patch_version": "16.8",
  "blue_champion_top": "Aatrox",
  "blue_champion_jgl": "Wukong",
  "blue_champion_mid": "Orianna",
  "blue_champion_bot": "KaiSa",
  "blue_champion_spt": "Rell",
  "red_champion_top": "Kennen",
  "red_champion_jgl": "Vi",
  "red_champion_mid": "Azir",
  "red_champion_bot": "Jhin",
  "red_champion_spt": "Nautilus",
  "blue_gold_top": 4300,
  "blue_gold_jgl": 3900,
  "blue_gold_mid": 4200,
  "blue_gold_bot": 4100,
  "blue_gold_spt": 2700,
  "red_gold_top": 4100,
  "red_gold_jgl": 3800,
  "red_gold_mid": 4000,
  "red_gold_bot": 3950,
  "red_gold_spt": 2600,
  "kills_left": 3,
  "kills_right": 1
}</textarea>
      </div>
    </section>

    <section class="card" style="margin-top: 18px;">
      <div class="label">Latest Payload</div>
      <pre id="payload">{}</pre>
    </section>
  </main>

  <script>
    let latestState = null;
    let selectedGameKey = null;
    try {
      selectedGameKey = window.localStorage.getItem("selectedLiveGameKey");
    } catch (error) {
      selectedGameKey = null;
    }

    function fmtGold(value) {
      if (value == null) return "-";
      return (value / 1000).toFixed(1) + "K";
    }

    function fmtGoldLead(value) {
      if (value == null) return "-";
      const prefix = value > 0 ? "+" : "";
      return prefix + (value / 1000).toFixed(1) + "K";
    }

    function fmtTime(value) {
      if (value == null) return "-";
      const mm = Math.floor(value / 60);
      const ss = String(value % 60).padStart(2, "0");
      return `${mm}:${ss}`;
    }

    function fmtPct(value) {
      if (value == null) return "-";
      return (value * 100).toFixed(1) + "%";
    }

    function fmtShortTime(value) {
      if (value == null) return "-";
      return fmtTime(value);
    }

    function isClockText(value) {
      return typeof value === "string" && /^\d{1,2}:\d{2}$/.test(value.trim());
    }

    function displayTimeValue(game) {
      if (!game) return "-";
      if (game.status !== "in_progress") {
        return "-";
      }
      return fmtTime(game.time_s);
    }

    function displayTimeRawValue(game) {
      if (!game) return "-";
      const raw = game.time_raw;
      if (typeof raw === "number") return fmtTime(raw);
      if (isClockText(raw)) return raw.trim();
      return "-";
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    function gameKey(game) {
      if (!game) return null;
      if (game.game_id != null) return `game:${game.game_id}`;
      if (game.match_id != null) return `match:${game.match_id}`;
      return null;
    }

    function setSelectedGameKey(value) {
      selectedGameKey = value || null;
      try {
        if (selectedGameKey) {
          window.localStorage.setItem("selectedLiveGameKey", selectedGameKey);
        } else {
          window.localStorage.removeItem("selectedLiveGameKey");
        }
      } catch (error) {
        // Ignore storage errors in restricted browsers.
      }
    }

    function chooseDisplayResult(result) {
      const liveGames = Array.isArray(result.live_games) ? result.live_games : [];
      if (!liveGames.length) return result;

      const selected = liveGames.find((game) => gameKey(game) === selectedGameKey);
      if (selected) return selected;

      const fallback = liveGames.find((game) => game.featured) || liveGames[0] || result;
      setSelectedGameKey(gameKey(fallback));
      return fallback;
    }

    function renderParticipants(rootId, players) {
      const root = document.getElementById(rootId);
      if (!players || players.length === 0) {
        root.innerHTML = '<div class="live-meta">No champion data yet.</div>';
        return;
      }

      root.innerHTML = players.map((player) => {
        const title = player.champion_id ?? player.summoner_name ?? "?";
        const metaParts = [player.role, player.summoner_name].filter(Boolean);
        return `
          <div class="player-row">
            <div class="player-main">
              <div class="player-name">${title}</div>
              <div class="player-meta">${metaParts.join(" • ") || "-"}</div>
            </div>
            <div class="player-gold">${fmtGold(player.total_gold)}</div>
          </div>
        `;
      }).join("");
    }

    function renderLiveGames(games, activeGameKey) {
      const root = document.getElementById("live-games");
      if (!games || games.length === 0) {
        root.innerHTML = '<div class="live-meta">No live matches found.</div>';
        return;
      }

      root.innerHTML = games.map((game) => {
        const itemKey = gameKey(game);
        const isSelected = itemKey != null && itemKey === activeGameKey;
        const teams = `${game.team_left ?? "?"} vs ${game.team_right ?? "?"}`;
        const league = game.league ?? "Unknown League";
        const gameLabel = game.game_number != null ? `Game ${game.game_number}` : "Game ?";
        const state = game.game_state ?? game.status ?? "unknown";
        const featured = game.featured ? " • featured" : "";
        const timeValue = displayTimeValue(game);
        const statusLine = game.status_message ? `<div class="live-meta" style="margin-top: 6px;">${game.status_message}</div>` : "";
        const statsLine = `
          <div class="live-stats">
            <div class="live-stat">
              <div class="live-stat-label">Gold</div>
              <div class="live-stat-value">${fmtGold(game.gold_left)} - ${fmtGold(game.gold_right)}</div>
            </div>
            <div class="live-stat">
              <div class="live-stat-label">Gold Lead</div>
              <div class="live-stat-value">${fmtGoldLead((game.gold_left != null && game.gold_right != null) ? (game.gold_left - game.gold_right) : null)}</div>
            </div>
            <div class="live-stat">
              <div class="live-stat-label">Time</div>
              <div class="live-stat-value">${timeValue}</div>
            </div>
          </div>
        `;
        const readyCount = game.prediction_comparison?.ready_model_count ?? 0;
        const oddsLine = readyCount
          ? `<div class="live-meta" style="margin-top: 8px;">Consensus blue ${escapeHtml(fmtPct(game.prediction_consensus_blue_win_prob))} • leader ${escapeHtml(game.prediction_comparison?.leader_model ?? "-")} • MLP ${escapeHtml(fmtPct(game.model_predictions?.mlp?.blue_win_prob))}</div>`
          : `<div class="live-meta" style="margin-top: 8px;">Odds unavailable: ${escapeHtml(game.prediction_reason ?? game.prediction_error ?? game.status_message ?? "-")}</div>`;
        const playersLeft = (game.participants_left || []).map((player) => `
          <div class="player-row">
            <div class="player-main">
              <div class="player-name">${player.champion_id ?? player.summoner_name ?? "?"}</div>
              <div class="player-meta">${[player.role, player.summoner_name].filter(Boolean).join(" • ") || "-"}</div>
            </div>
            <div class="player-gold">${fmtGold(player.total_gold)}</div>
          </div>
        `).join("");
        const playersRight = (game.participants_right || []).map((player) => `
          <div class="player-row">
            <div class="player-main">
              <div class="player-name">${player.champion_id ?? player.summoner_name ?? "?"}</div>
              <div class="player-meta">${[player.role, player.summoner_name].filter(Boolean).join(" • ") || "-"}</div>
            </div>
            <div class="player-gold">${fmtGold(player.total_gold)}</div>
          </div>
        `).join("");
        const playersBlock = (playersLeft || playersRight) ? `
          <div class="players" style="margin-top: 12px;">
            <div class="player-list">${playersLeft || '<div class="live-meta">No champion data yet.</div>'}</div>
            <div class="player-list">${playersRight || '<div class="live-meta">No champion data yet.</div>'}</div>
          </div>
        ` : "";
        return `
          <div class="live-item${game.featured ? " featured" : ""}${isSelected ? " selected" : ""}" data-game-key="${escapeHtml(itemKey ?? "")}">
            <strong>${teams}</strong>
            <div class="live-meta">${league} • ${gameLabel} • ${state}${featured}</div>
            ${statusLine}
            ${statsLine}
            ${oddsLine}
            ${playersBlock}
          </div>
        `;
      }).join("");

      root.querySelectorAll(".live-item[data-game-key]").forEach((node) => {
        node.addEventListener("click", () => {
          const nextKey = node.dataset.gameKey || null;
          if (!nextKey || nextKey === selectedGameKey) return;
          setSelectedGameKey(nextKey);
          if (latestState) renderState(latestState);
        });
      });
    }

    function renderModelPredictions(result) {
      const root = document.getElementById("model-grid");
      const predictions = result.model_predictions || {};
      const order = ["gru", "logistic_regression", "xgboost", "mlp"];
      const primary = result.prediction_model;
      root.innerHTML = order.map((key) => {
        const entry = predictions[key] || {};
        const ready = entry.status === "ready";
        const meta = ready
          ? (key === "gru" ? "live sequence" : "ready")
          : `not ready: ${entry.reason ?? entry.status ?? "-"}`;
        return `
          <div class="model-card${primary === key ? " primary" : ""}">
            <div class="model-card-head">
              <div class="model-card-title">${escapeHtml(entry.model_label ?? key)}</div>
              <div class="model-card-state">${escapeHtml(entry.status ?? "-")}</div>
            </div>
            <div class="model-card-values">
              <span>${fmtPct(entry.blue_win_prob)}</span>
              <span>${fmtPct(entry.red_win_prob)}</span>
            </div>
            <div class="model-card-meta">${escapeHtml(meta)}</div>
          </div>
        `;
      }).join("");
    }

    function renderPredictionHistory(result) {
      const rows = result.prediction_history_preview || [];
      const body = document.getElementById("history-body");
      const meta = document.getElementById("history-meta");
      const path = result.live_log_path || "-";
      meta.textContent = rows.length
        ? `Showing the latest ${rows.length} saved rows from ${path}`
        : `No saved rows yet. Current log: ${path}`;
      if (!rows.length) {
        body.innerHTML = '<tr><td colspan="6" class="live-meta">No history yet.</td></tr>';
        return;
      }

      body.innerHTML = rows.map((row) => {
        const models = row.model_predictions || {};
        return `
          <tr>
            <td>${escapeHtml(fmtShortTime(row.time_s))}</td>
            <td>${escapeHtml(fmtPct(row.prediction_consensus_blue_win_prob))}</td>
            <td>${escapeHtml(fmtPct(models.gru?.blue_win_prob))}</td>
            <td>${escapeHtml(fmtPct(models.logistic_regression?.blue_win_prob))}</td>
            <td>${escapeHtml(fmtPct(models.xgboost?.blue_win_prob))}</td>
            <td>${escapeHtml(fmtPct(models.mlp?.blue_win_prob))}</td>
          </tr>
        `;
      }).join("");
    }

    function buildManualTemplate(result) {
      const base = {
        source: "manual",
        status: result.status ?? "in_progress",
        league: result.league ?? "",
        source_season: result.source_season ?? "",
        tournament: result.tournament ?? result.league ?? "",
        tournament_stage: result.tournament_stage ?? "",
        match_id: result.match_id ?? "manual_match_1",
        game_id: result.game_id ?? 900001,
        game_number: result.game_number ?? 1,
        team_left: result.team_left ?? "Blue Team",
        team_right: result.team_right ?? "Red Team",
        time_s: result.time_s ?? 600,
        patch_version: result.patch_version ?? "",
        gold_left: result.gold_left ?? null,
        gold_right: result.gold_right ?? null,
        kills_left: result.kills_left ?? 0,
        kills_right: result.kills_right ?? 0
      };
      ["top", "jgl", "mid", "bot", "spt"].forEach((role) => {
        base[`blue_champion_${role}`] = result[`blue_champion_${role}`] ?? "";
        base[`red_champion_${role}`] = result[`red_champion_${role}`] ?? "";
        base[`blue_gold_${role}`] = result[`blue_gold_${role}`] ?? null;
        base[`red_gold_${role}`] = result[`red_gold_${role}`] ?? null;
      });
      return base;
    }

    async function applyManualState() {
      const raw = document.getElementById("manual-json").value;
      let payload;
      try {
        payload = JSON.parse(raw);
      } catch (error) {
        document.getElementById("manual-meta").textContent = `Manual JSON error: ${error}`;
        return;
      }

      const res = await fetch("/api/manual-state", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      document.getElementById("manual-meta").textContent = data.error
        ? `Manual override failed: ${data.error}`
        : "Manual override active.";
      await refresh();
    }

    async function clearManualState() {
      await fetch("/api/manual-state", { method: "DELETE" });
      document.getElementById("manual-meta").textContent = "Manual override cleared.";
      await refresh();
    }

    async function replayTrainingFile() {
      const path = document.getElementById("replay-file-path").value.trim();
      const gameId = document.getElementById("replay-game-id").value.trim();
      if (!path) {
        document.getElementById("manual-meta").textContent = "Replay file path is required.";
        return;
      }
      const res = await fetch("/api/replay-file", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path, game_id: gameId || null })
      });
      const data = await res.json();
      if (data.error) {
        document.getElementById("manual-meta").textContent = `Replay failed: ${data.error}`;
        return;
      }
      const scope = gameId ? `game ${gameId}` : "all matching games";
      document.getElementById("manual-meta").textContent =
        `Replay complete for ${scope}. Games processed: ${data.games_processed}. Rows written: ${data.rows_written}. Summary: ${data.summary_path ?? "-"}`;
      await refresh();
    }

    function renderState(data) {
      latestState = data;
      const rootResult = data.result || {};
      const r = chooseDisplayResult(rootResult);
      const status = document.getElementById("status-text");
      const matchup = document.getElementById("matchup");
      if (r.team_left || r.team_right) {
        matchup.textContent = `${r.team_left ?? "?"} vs ${r.team_right ?? "?"}`;
      } else {
        matchup.textContent = "-";
      }
      if (data.error) {
        status.innerHTML = `<span class="error">${data.error}</span>`;
      } else if (data.result) {
        const updated = data.updated_at ? new Date(data.updated_at).toLocaleTimeString() : "never";
        const source = (rootResult.source || r.source) ? ` via ${(rootResult.source || r.source).toUpperCase()}` : "";
        if (r.manual_override) {
          status.textContent = `Manual snapshot active${source}, updated ${updated}`;
        } else if (r.status_message && r.status !== "in_progress") {
          status.textContent = `${r.status_message}${source}, updated ${updated}`;
        } else {
          status.textContent = `Scanner running${source}, updated ${updated}`;
        }
      } else {
        status.textContent = "Waiting for first frame…";
      }

      document.getElementById("gold-left").textContent = fmtGold(r.gold_left);
      document.getElementById("gold-right").textContent = fmtGold(r.gold_right);
      document.getElementById("gold-left-raw").textContent = `raw: ${r.gold_left_raw ?? "-"}`;
      document.getElementById("gold-right-raw").textContent = `raw: ${r.gold_right_raw ?? "-"}`;
      const goldDiff = (r.gold_left != null && r.gold_right != null) ? (r.gold_left - r.gold_right) : null;
      document.getElementById("gold-diff").textContent = fmtGoldLead(goldDiff);
      document.getElementById("gold-diff-raw").textContent =
        goldDiff == null
          ? "derived from team gold totals"
          : `blue ${fmtGold(r.gold_left)} vs red ${fmtGold(r.gold_right)}`;
      document.getElementById("time").textContent = displayTimeValue(r);
      document.getElementById("time-raw").textContent =
        `raw: ${displayTimeRawValue(r)}${r.time_source ? ` (${r.time_source})` : ""}` +
        ((r.time_sync_offset_s ?? 0) > 0 ? ` | synced -${r.time_sync_offset_s}s to api gold` : "");
      document.getElementById("pred-spread").textContent = fmtPct(r.prediction_comparison?.spread_blue_win_prob);
      const readyModelCount = r.prediction_comparison?.ready_model_count ?? 0;
      const totalModelCount = r.prediction_comparison?.total_model_count ?? 4;
      const leaderModel = r.prediction_comparison?.leader_model ?? "-";
      document.getElementById("pred-spread-raw").textContent =
        readyModelCount
          ? `leader: ${leaderModel}`
          : "largest gap across model predictions";
      document.getElementById("pred-blue").textContent = fmtPct(r.prediction_consensus_blue_win_prob);
      document.getElementById("pred-red").textContent = fmtPct(r.prediction_consensus_red_win_prob);
      document.getElementById("pred-meta").textContent =
        readyModelCount
          ? "average of available model predictions"
          : `not ready: ${r.prediction_reason ?? r.prediction_error ?? "-"}`;
      document.getElementById("pred-steps").textContent =
        `primary: ${r.prediction_model ?? "-"} | current minute: ${r.prediction_current_minute ?? "-"} | leader: ${leaderModel}`;
      document.getElementById("model-grid-meta").textContent =
        readyModelCount
          ? `models ready: ${readyModelCount}/${totalModelCount} | leader: ${leaderModel} | spread: ${fmtPct(r.prediction_comparison?.spread_blue_win_prob)}`
          : `models ready: 0/${totalModelCount} | waiting for predictions`;
      renderModelPredictions(r);
      renderParticipants("players-left", r.participants_left || []);
      renderParticipants("players-right", r.participants_right || []);
      renderLiveGames(rootResult.live_games || [], gameKey(r));
      renderPredictionHistory(r);
      document.getElementById("payload").textContent = JSON.stringify(data, null, 2);
    }

    async function refresh() {
      const res = await fetch("/api/state", { cache: "no-store" });
      const data = await res.json();
      renderState(data);
    }

    document.getElementById("manual-template-btn").addEventListener("click", () => {
      try {
        const current = chooseDisplayResult((latestState && latestState.result) || {});
        document.getElementById("manual-json").value = JSON.stringify(buildManualTemplate(current), null, 2);
        document.getElementById("manual-meta").textContent = "Current payload copied into the manual editor.";
      } catch (error) {
        document.getElementById("manual-meta").textContent = `Could not build template: ${error}`;
      }
    });
    document.getElementById("manual-apply-btn").addEventListener("click", applyManualState);
    document.getElementById("manual-clear-btn").addEventListener("click", clearManualState);
    document.getElementById("replay-file-btn").addEventListener("click", replayTrainingFile);

    refresh();
    setInterval(refresh, 5000);
    window.addEventListener("focus", refresh);
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) refresh();
    });
  </script>
</body>
</html>
"""
INDEX_HTML = INDEX_HTML.replace(
    'value="data/training_table_all.csv"',
    f'value="{DEFAULT_REPLAY_CSV_PATH}"',
)


class LiveScanState:
    def __init__(self):
        self._lock = threading.Lock()
        self.result = None
        self.error = None
        self.updated_at = None

    def set_result(self, result: dict):
        with self._lock:
            self.result = result
            self.error = None
            self.updated_at = datetime.now(timezone.utc).isoformat()

    def set_error(self, error: str):
        with self._lock:
            self.error = error
            self.updated_at = datetime.now(timezone.utc).isoformat()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "result": self.result,
                "error": self.error,
                "updated_at": self.updated_at,
            }


class ManualOverrideState:
    def __init__(self):
        self._lock = threading.Lock()
        self.result = None
        self.updated_at = None

    def set_result(self, result: dict):
        with self._lock:
            self.result = result
            self.updated_at = datetime.now(timezone.utc).isoformat()

    def clear(self):
        with self._lock:
            self.result = None
            self.updated_at = datetime.now(timezone.utc).isoformat()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "result": self.result,
                "updated_at": self.updated_at,
                "active": self.result is not None,
            }


def slugify_log_part(value) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def build_live_log_row(result: dict, captured_at: datetime) -> dict:
    row = {
        "captured_at": captured_at.isoformat(),
        "source": result.get("source"),
        "manual_override": bool(result.get("manual_override")),
        "status": result.get("status"),
        "status_message": result.get("status_message"),
        "league": result.get("league"),
        "tournament": result.get("tournament"),
        "tournament_stage": result.get("tournament_stage"),
        "source_season": result.get("source_season"),
        "source_file": result.get("source_file"),
        "match_id": result.get("match_id"),
        "game_id": result.get("game_id"),
        "game_number": result.get("game_number"),
        "game_state": result.get("game_state"),
        "team_left": result.get("team_left"),
        "team_right": result.get("team_right"),
        "time_s": result.get("time_s"),
        "time_raw": result.get("time_raw"),
        "time_source": result.get("time_source"),
        "time_ocr_raw": result.get("time_ocr_raw"),
        "time_ocr_error": result.get("time_ocr_error"),
        "time_sync_offset_s": result.get("time_sync_offset_s"),
        "feed_age_s": result.get("feed_age_s"),
        "gold_left": result.get("gold_left"),
        "gold_right": result.get("gold_right"),
        "gold_diff_blue": (
            result.get("gold_left") - result.get("gold_right")
            if result.get("gold_left") is not None and result.get("gold_right") is not None
            else None
        ),
        "kills_left": result.get("kills_left"),
        "kills_right": result.get("kills_right"),
        "patch_version": result.get("patch_version"),
        "prediction_status": result.get("prediction_status"),
        "prediction_reason": result.get("prediction_reason"),
        "prediction_model": result.get("prediction_model"),
        "prediction_prefix_minute": result.get("prediction_prefix_minute"),
        "prediction_timesteps": result.get("prediction_timesteps"),
        "prediction_current_minute": result.get("prediction_current_minute"),
        "prediction_blue_win_prob": result.get("prediction_blue_win_prob"),
        "prediction_red_win_prob": result.get("prediction_red_win_prob"),
        "prediction_consensus_blue_win_prob": result.get("prediction_consensus_blue_win_prob"),
        "prediction_consensus_red_win_prob": result.get("prediction_consensus_red_win_prob"),
        "prediction_comparison": result.get("prediction_comparison"),
        "model_predictions": result.get("model_predictions"),
        "winner_side": result.get("winner_side"),
        "winner_team": result.get("winner_team"),
        "blue_win": result.get("blue_win"),
    }
    for role in ROLE_ORDER:
        blue_gold = result.get(f"blue_gold_{role}")
        red_gold = result.get(f"red_gold_{role}")
        row[f"blue_champion_{role}"] = result.get(f"blue_champion_{role}")
        row[f"red_champion_{role}"] = result.get(f"red_champion_{role}")
        row[f"blue_gold_{role}"] = blue_gold
        row[f"red_gold_{role}"] = red_gold
        row[f"gold_lead_{role}"] = (
            blue_gold - red_gold
            if blue_gold is not None and red_gold is not None
            else None
        )
    return row


def parse_time_input(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return max(0, int(value))
    text = str(value).strip()
    if not text:
        return None
    if ":" not in text:
        try:
            return max(0, int(float(text)))
        except ValueError:
            return None
    parts = text.split(":")
    if len(parts) != 2:
        return None
    try:
        minutes = int(parts[0])
        seconds = int(parts[1])
    except ValueError:
        return None
    if minutes < 0 or seconds < 0 or seconds >= 60:
        return None
    return minutes * 60 + seconds


def format_time_clock(value) -> str | None:
    parsed = parse_time_input(value)
    if parsed is None:
        return None
    return f"{parsed // 60}:{parsed % 60:02d}"


def sync_timer_to_feed_age(result: dict, timer_s: int) -> tuple[int, int]:
    if result.get("source") != "api":
        return timer_s, 0

    try:
        feed_age_s = int(result.get("feed_age_s"))
    except (TypeError, ValueError):
        return timer_s, 0

    if feed_age_s <= 0:
        return timer_s, 0
    return max(0, timer_s - feed_age_s), feed_age_s


def merge_ocr_scoreboard(result: dict, scan_result: dict | None) -> dict:
    if not isinstance(scan_result, dict):
        return result

    merged = dict(result)
    for key in (
        "gold_left_raw",
        "gold_right_raw",
        "kills_left_raw",
        "kills_right_raw",
        "roi_dir",
    ):
        value = scan_result.get(key)
        if value is not None:
            merged[key] = value

    for key in ("gold_left", "gold_right", "kills_left", "kills_right"):
        value = scan_result.get(key)
        if value is not None:
            merged[key] = value
    return merged


def coerce_game_id(value, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        digits = re.sub(r"\D+", "", str(value))
        return int(digits) if digits else default


def hydrate_manual_totals(result: dict) -> dict:
    result = dict(result)
    blue_total = 0
    red_total = 0
    blue_complete = True
    red_complete = True
    for role in ROLE_ORDER:
        blue_gold = result.get(f"blue_gold_{role}")
        red_gold = result.get(f"red_gold_{role}")
        if blue_gold is None:
            blue_complete = False
        else:
            blue_total += int(blue_gold)
        if red_gold is None:
            red_complete = False
        else:
            red_total += int(red_gold)
    if result.get("gold_left") is None and blue_complete:
        result["gold_left"] = blue_total
    if result.get("gold_right") is None and red_complete:
        result["gold_right"] = red_total
    if result.get("kills_left") is None:
        result["kills_left"] = 0
    if result.get("kills_right") is None:
        result["kills_right"] = 0
    return result


def build_manual_participants(result: dict, side: str) -> list[dict]:
    players: list[dict] = []
    team_key = "team_left" if side == "blue" else "team_right"
    team_name = str(result.get(team_key) or "").strip()
    team_abbrev = "".join(part[:1] for part in team_name.split()).upper()
    for role in ROLE_ORDER:
        champion = result.get(f"{side}_champion_{role}")
        gold = result.get(f"{side}_gold_{role}")
        summoner = (
            result.get(f"{side}_player_{role}")
            or result.get(f"{side}_summoner_{role}")
            or (f"{team_abbrev} {role.upper()}" if team_abbrev else None)
        )
        if champion is None and gold is None and summoner is None:
            continue
        players.append(
            {
                "role": role,
                "champion_id": champion,
                "summoner_name": summoner,
                "total_gold": gold,
            }
        )
    return players


def load_prediction_history_preview(path: str | None, limit: int = 60) -> list[dict]:
    if not path:
        return []
    file_path = Path(path)
    if not file_path.exists():
        return []

    rows: deque[str] = deque(maxlen=limit)
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(line)

    preview: list[dict] = []
    for raw in rows:
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        preview.append(
            {
                "captured_at": row.get("captured_at"),
                "time_s": row.get("time_s"),
                "status": row.get("status"),
                "prediction_blue_win_prob": row.get("prediction_blue_win_prob"),
                "prediction_consensus_blue_win_prob": row.get("prediction_consensus_blue_win_prob"),
                "model_predictions": row.get("model_predictions") or {},
            }
        )
    return preview


def role_name_to_live_key(role: str) -> str | None:
    mapping = {
        "TOP": "top",
        "JGL": "jgl",
        "MID": "mid",
        "BOT": "bot",
        "SPT": "spt",
    }
    return mapping.get(str(role or "").strip().upper())


def load_training_style_games(csv_path: str, target_game_id: str | None = None) -> dict[str, dict]:
    games: dict[str, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            game_id = str(row.get("game_id") or "").strip()
            if not game_id:
                continue
            if target_game_id and game_id != target_game_id:
                continue
            minute_text = str(row.get("minute") or "").strip()
            if not minute_text:
                continue
            try:
                minute = int(float(minute_text))
            except ValueError:
                continue
            role_key = role_name_to_live_key(row.get("role") or "")
            if role_key is None:
                continue

            game = games.setdefault(
                game_id,
                {
                    "base": row,
                    "minutes": {},
                },
            )
            minute_rows = game["minutes"].setdefault(minute, {})
            minute_rows[role_key] = row
    return games


def build_replay_snapshot_from_rows(game_id: str, minute: int, rows_by_role: dict[str, dict[str, str]]) -> dict:
    base_row = next(iter(rows_by_role.values()))
    result = {
        "source": "manual",
        "manual_override": True,
        "status": "in_progress",
        "status_message": f"Replay snapshot from {Path(base_row.get('source_file') or '').name or 'training file'}",
        "source_file": base_row.get("source_file"),
        "source_season": base_row.get("source_season"),
        "league": base_row.get("league"),
        "tournament": base_row.get("tournament"),
        "tournament_stage": base_row.get("tournament_stage"),
        "match_id": f"replay_{Path(base_row.get('source_file') or 'file').stem}_{game_id}",
        "game_id": coerce_game_id(game_id, default=900001),
        "game_number": 1,
        "game_state": "replay",
        "team_left": base_row.get("blue_team"),
        "team_right": base_row.get("red_team"),
        "time_s": minute * 60,
        "time_source": "replay_file",
        "time_raw": f"{minute}:00",
        "patch_version": base_row.get("patch"),
        "winner_side": base_row.get("winner_side"),
        "winner_team": base_row.get("winner_team"),
        "blue_win": int(base_row.get("blue_win") or 0),
        "date": base_row.get("date"),
        "live_games": [],
    }
    for role in ROLE_ORDER:
        row = rows_by_role.get(role)
        if row is None:
            continue
        result[f"blue_champion_{role}"] = row.get("blue_champion")
        result[f"red_champion_{role}"] = row.get("red_champion")
        result[f"blue_gold_{role}"] = int(float(row.get("blue_gold") or 0))
        result[f"red_gold_{role}"] = int(float(row.get("red_gold") or 0))
    result = hydrate_manual_totals(result)
    result["participants_left"] = build_manual_participants(result, "blue")
    result["participants_right"] = build_manual_participants(result, "red")
    return result


def build_replay_summary_rows(results: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for result in results:
        model_predictions = result.get("model_predictions") or {}
        row = {
            "source_file": result.get("source_file"),
            "game_id": result.get("game_id"),
            "date": result.get("date"),
            "league": result.get("league"),
            "tournament": result.get("tournament"),
            "tournament_stage": result.get("tournament_stage"),
            "team_left": result.get("team_left"),
            "team_right": result.get("team_right"),
            "minute": int((result.get("time_s") or 0) // 60),
            "time_s": result.get("time_s"),
            "winner_side": result.get("winner_side"),
            "winner_team": result.get("winner_team"),
            "blue_win": result.get("blue_win"),
            "consensus_blue_win_prob": result.get("prediction_consensus_blue_win_prob"),
        }
        for key in ("gru", "logistic_regression", "xgboost", "mlp"):
            entry = model_predictions.get(key) or {}
            row[f"{key}_blue_win_prob"] = entry.get("blue_win_prob")
            row[f"{key}_status"] = entry.get("status")
        rows.append(row)
    return rows


def write_replay_summary_csv(rows: list[dict], output_dir: str, csv_path: str, game_id: str | None) -> str | None:
    if not rows:
        return None
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    file_part = slugify_log_part(Path(csv_path).stem)
    game_part = slugify_log_part(game_id or "all_games")
    output_path = output_root / f"{stamp}_{file_part}_{game_part}_model_probs.csv"
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(output_path)


def create_app(
    source: str = "ocr",
    channel: str = "riotgames",
    quality: str = "best",
    profile_path: str = DEFAULT_PROFILE_PATH,
    interval: float = 5.0,
    timer_source: str = "match-stream",
    timer_url: str = "https://lolesports.com/live",
    timer_headful: bool = False,
    live_log_dir: str | None = DEFAULT_LIVE_LOG_DIR,
    replay_output_dir: str = DEFAULT_REPLAY_OUTPUT_DIR,
):
    app = Flask(__name__)

    def no_cache(response: Response):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    state = LiveScanState()
    manual_state = ManualOverrideState()
    profile = load_profile(profile_path) if source in {"ocr", "api"} else None
    profile_cache: dict[str, dict] = {}
    api_client = LolesportsLiveClient() if source == "api" else None
    predictor = LiveSequencePredictor()
    use_live_page_browser = source == "api" and (timer_source == "lolesports" or not api_is_configured())
    timer_browser = (
        LolesportsTimerOCR(
            profile_path=profile_path,
            url=timer_url,
            headless=not timer_headful,
        )
        if use_live_page_browser
        else None
    )
    stop_event = threading.Event()
    last_timer_by_game: dict[object, tuple[int, float]] = {}
    last_log_key_by_game: dict[object, tuple[object, ...]] = {}
    last_log_path_by_game: dict[object, str] = {}
    last_logged_minute_by_game: dict[object, int | None] = {}
    predictor_state_by_game: dict[object, dict[str, object]] = {}

    def load_cached_profile(path: str) -> dict:
        cached = profile_cache.get(path)
        if cached is not None:
            return cached
        loaded = load_profile(path)
        profile_cache[path] = loaded
        return loaded

    def pick_timer_profile(result: dict | None) -> dict | None:
        if source not in {"ocr", "api"}:
            return None
        league = str((result or {}).get("league") or "").strip().upper()
        if league == "LCK":
            return load_cached_profile(DEFAULT_LCK_PROFILE_PATH)
        return profile

    def merge_live_page_match_info(result: dict, page_info: dict) -> dict:
        merged = dict(result)
        merged["live_page_url"] = page_info.get("page_url")
        merged["live_page_error"] = page_info.get("error")

        for key in ("league", "team_left", "team_right", "match_id"):
            if not merged.get(key) and page_info.get(key):
                merged[key] = page_info.get(key)

        if merged.get("status") in {"api_unavailable", "no_live_event"} or not merged.get("team_left"):
            merged["source"] = page_info.get("source") or merged.get("source")
            merged["status"] = page_info.get("status") or merged.get("status")
            merged["status_message"] = page_info.get("status_message") or merged.get("status_message")
            if page_info.get("match_id"):
                merged["match_id"] = page_info.get("match_id")
        return merged
    log_root = Path(live_log_dir) if live_log_dir else None

    def result_game_key(result: dict) -> object | None:
        if not isinstance(result, dict):
            return None
        return result.get("game_id") or result.get("match_id")

    def same_game(left: dict, right: dict) -> bool:
        if not isinstance(left, dict) or not isinstance(right, dict):
            return False
        left_game_id = left.get("game_id")
        right_game_id = right.get("game_id")
        if left_game_id is not None and right_game_id is not None:
            return left_game_id == right_game_id
        left_match_id = left.get("match_id")
        right_match_id = right.get("match_id")
        return left_match_id is not None and left_match_id == right_match_id

    def snapshot_predictor_state() -> dict[str, object]:
        return {
            "current_game_id": predictor.current_game_id,
            "minute_history": {
                int(minute): dict(values)
                for minute, values in predictor.minute_history.items()
            },
        }

    def restore_predictor_state(game_key: object | None) -> None:
        if game_key is None:
            predictor.reset()
            return
        saved = predictor_state_by_game.get(game_key)
        if saved is None:
            predictor.reset()
            return
        predictor.current_game_id = saved.get("current_game_id")
        minute_history = saved.get("minute_history") or {}
        predictor.minute_history = {
            int(minute): dict(values)
            for minute, values in minute_history.items()
            if isinstance(values, dict)
        }

    def enrich_result_with_predictor(result: dict) -> dict:
        if predictor is None or not isinstance(result, dict):
            return result
        game_key = result_game_key(result)
        restore_predictor_state(game_key)
        enriched = predictor.enrich(result)
        if game_key is not None:
            predictor_state_by_game[game_key] = snapshot_predictor_state()
        return enriched

    def compact_live_game_entry(result: dict) -> dict:
        entry = dict(result)
        entry.pop("live_games", None)
        return entry

    def append_live_log(result: dict) -> dict:
        if log_root is None:
            return result
        if not isinstance(result, dict):
            return result
        result_source = str(result.get("source") or source).lower()
        if result_source not in {"api", "manual"}:
            return result
        if result.get("match_id") is None and result.get("game_id") is None:
            return result

        game_key = result.get("game_id") or result.get("match_id")
        minute_bucket = (
            int(result["time_s"]) // 60
            if result.get("time_s") is not None
            else None
        )
        log_key = (
            minute_bucket,
            result.get("status"),
            result.get("prediction_status"),
            result.get("prediction_reason"),
        )
        if game_key is not None and last_log_key_by_game.get(game_key) == log_key:
            result = dict(result)
            existing_path = last_log_path_by_game.get(game_key)
            result["live_log_path"] = existing_path
            result["live_log_skipped"] = True
            result["prediction_history_preview"] = load_prediction_history_preview(existing_path)
            return result

        captured_at = datetime.now(timezone.utc)
        match_part = slugify_log_part(result.get("match_id") or "match")
        game_part = slugify_log_part(result.get("game_id") or "game")
        teams_part = slugify_log_part(
            f"{result.get('team_left') or 'blue'}_vs_{result.get('team_right') or 'red'}"
        )

        path_str = last_log_path_by_game.get(game_key) if game_key is not None else None
        last_logged_minute = (
            last_logged_minute_by_game.get(game_key) if game_key is not None else None
        )
        start_new_log = path_str is None
        if (
            game_key is not None
            and minute_bucket is not None
            and last_logged_minute is not None
            and minute_bucket < last_logged_minute
        ):
            start_new_log = True

        log_root.mkdir(parents=True, exist_ok=True)
        if start_new_log:
            filename = (
                f"{captured_at:%Y%m%dT%H%M%S}_{match_part}_{game_part}_{teams_part}.jsonl"
            )
            path = log_root / filename
        else:
            path = Path(path_str)

        row = build_live_log_row(result, captured_at)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

        result = dict(result)
        if game_key is not None:
            last_log_key_by_game[game_key] = log_key
            last_log_path_by_game[game_key] = str(path)
            last_logged_minute_by_game[game_key] = minute_bucket
        result["live_log_path"] = str(path)
        result["live_log_skipped"] = False
        result["prediction_history_preview"] = load_prediction_history_preview(str(path))
        return result

    def enrich_live_games(result: dict) -> dict:
        if predictor is None or not isinstance(result, dict):
            return result
        live_games = result.get("live_games")
        if not isinstance(live_games, list):
            return result

        enriched_games: list[dict] = []
        changed = False
        for game in live_games:
            if not isinstance(game, dict):
                enriched_games.append(game)
                continue
            if same_game(game, result):
                featured_game = compact_live_game_entry(result)
                featured_game["featured"] = bool(game.get("featured") or result.get("featured"))
                enriched_games.append(featured_game)
                changed = True
                continue

            enriched_game = enrich_result_with_predictor(game)
            enriched_game = append_live_log(enriched_game)
            enriched_game["featured"] = bool(game.get("featured"))
            enriched_games.append(compact_live_game_entry(enriched_game))
            changed = True

        if not changed:
            return result
        result = dict(result)
        result["live_games"] = enriched_games
        return result

    def prepare_manual_result(payload: dict) -> dict:
        result = dict(payload)
        result["source"] = "manual"
        result["manual_override"] = True
        result.setdefault("status", "in_progress")
        result.setdefault("status_message", "Manual snapshot active.")
        result.setdefault("league", result.get("league") or "Manual")
        result.setdefault("source_season", result.get("source_season") or "")
        result.setdefault("match_id", result.get("match_id") or "manual_match_1")
        result["game_id"] = coerce_game_id(result.get("game_id"), default=900001)
        parsed_time = parse_time_input(result.get("time_s") or result.get("time"))
        if parsed_time is None and result.get("minute") is not None:
            try:
                parsed_time = max(0, int(float(result.get("minute"))) * 60)
            except (TypeError, ValueError):
                parsed_time = None
        result["time_s"] = parsed_time or 0
        result.setdefault("game_number", 1)
        result.setdefault("team_left", result.get("team_left") or "Blue Team")
        result.setdefault("team_right", result.get("team_right") or "Red Team")
        result.setdefault("game_state", result.get("game_state") or "manual")
        result.setdefault("live_games", [])
        result.setdefault("time_source", "manual")
        result.setdefault("time_raw", result.get("time_s"))
        result = hydrate_manual_totals(result)
        result["participants_left"] = build_manual_participants(result, "blue")
        result["participants_right"] = build_manual_participants(result, "red")
        if predictor is not None:
            result = enrich_result_with_predictor(result)
        result = append_live_log(result)
        return result

    def run_replay_file(csv_path: str, target_game_id: str | None = None) -> dict:
        manual_state.clear()
        predictor_state_by_game.clear()
        games = load_training_style_games(csv_path, target_game_id=target_game_id)
        if not games:
            raise ValueError("No matching games found in the provided file.")

        replay_results: list[dict] = []
        processed_games = 0
        last_result = None
        for game_id, game in sorted(games.items(), key=lambda item: item[0]):
            minute_map = game["minutes"]
            usable_minutes = [
                minute
                for minute, rows_by_role in sorted(minute_map.items())
                if all(role in rows_by_role for role in ROLE_ORDER)
            ]
            if not usable_minutes:
                continue
            predictor.reset()
            processed_games += 1
            for minute in usable_minutes:
                snapshot = build_replay_snapshot_from_rows(game_id, minute, minute_map[minute])
                snapshot = enrich_result_with_predictor(snapshot)
                snapshot = append_live_log(snapshot)
                replay_results.append(snapshot)
                last_result = snapshot

        if not replay_results:
            raise ValueError("Found games, but none had complete five-role minute snapshots.")

        summary_rows = build_replay_summary_rows(replay_results)
        summary_path = write_replay_summary_csv(
            summary_rows,
            output_dir=replay_output_dir,
            csv_path=csv_path,
            game_id=target_game_id,
        )

        if target_game_id and last_result is not None:
            last_result = dict(last_result)
            last_result["status_message"] = f"Replay complete from {Path(csv_path).name}"
            last_result["replay_summary_path"] = summary_path
            manual_state.set_result(last_result)

        return {
            "games_processed": processed_games,
            "rows_written": len(summary_rows),
            "summary_path": summary_path,
            "last_result": last_result,
        }

    def apply_timer_result(
        result: dict,
        time_s,
        time_raw,
        source_name: str,
        error: str | None = None,
    ) -> dict:
        result = dict(result)
        existing_time_s = result.get("time_s")
        existing_time_raw = result.get("time_raw")
        existing_time_source = result.get("time_source")
        result["time_ocr_raw"] = time_raw
        result["time_sync_offset_s"] = 0
        game_key = result.get("game_id") or result.get("match_id")
        now = time.time()

        def normalize_time_raw(value):
            if parse_time_input(value) is None:
                return None
            return format_time_clock(value) or str(value).strip()

        existing_time_raw = normalize_time_raw(existing_time_raw)
        normalized_time_raw = normalize_time_raw(time_raw)

        def use_holdover(reason: str) -> dict:
            previous = last_timer_by_game.get(game_key)
            if previous is not None and now - previous[1] <= max(30.0, interval * 3):
                result["time_s"] = previous[0] + int(now - previous[1])
                result["time_raw"] = normalized_time_raw or existing_time_raw
                result["time_source"] = "timer_holdover"
            elif existing_time_s is not None:
                result["time_s"] = existing_time_s
                result["time_raw"] = existing_time_raw
                result["time_source"] = existing_time_source or "api"
            else:
                result["time_s"] = None
                result["time_raw"] = None
                result["time_source"] = "timer_unavailable"
            result["time_ocr_error"] = reason
            return result

        if time_s is None:
            return use_holdover(error or "Timer OCR returned no valid time.")

        try:
            timer_s = int(time_s)
        except (TypeError, ValueError):
            return use_holdover(f"Timer OCR returned invalid time: {time_s!r}")

        if timer_s < 0 or timer_s > 90 * 60:
            return use_holdover(f"Timer OCR returned implausible time: {time_raw or timer_s}")

        timer_s, sync_offset_s = sync_timer_to_feed_age(result, timer_s)

        previous = last_timer_by_game.get(game_key)
        if previous is not None and timer_s + 20 < previous[0]:
            return use_holdover(
                f"Timer OCR moved backward from {previous[0]}s to {timer_s}s."
            )

        last_timer_by_game[game_key] = (timer_s, now)
        result["time_s"] = timer_s
        result["time_raw"] = format_time_clock(timer_s)
        result["time_source"] = source_name
        result["time_ocr_error"] = error
        result["time_sync_offset_s"] = sync_offset_s
        return result

    def strip_interface_prediction_metadata(result: dict | None) -> dict | None:
        if not isinstance(result, dict):
            return result

        cleaned = dict(result)
        cleaned.pop("prediction_prefix_minute", None)
        cleaned.pop("prediction_timesteps", None)

        predictions = cleaned.get("model_predictions")
        if isinstance(predictions, dict):
            cleaned["model_predictions"] = {
                key: (
                    {
                        sub_key: sub_value
                        for sub_key, sub_value in value.items()
                        if sub_key not in {"prefix_minute", "timesteps", "artifact_mode"}
                    }
                    if isinstance(value, dict)
                    else value
                )
                for key, value in predictions.items()
            }

        live_games = cleaned.get("live_games")
        if isinstance(live_games, list):
            cleaned["live_games"] = [
                strip_interface_prediction_metadata(game) if isinstance(game, dict) else game
                for game in live_games
            ]

        history_rows = cleaned.get("prediction_history_preview")
        if isinstance(history_rows, list):
            cleaned_history: list[object] = []
            for row in history_rows:
                if not isinstance(row, dict):
                    cleaned_history.append(row)
                    continue
                history_entry = dict(row)
                model_predictions = history_entry.get("model_predictions")
                if isinstance(model_predictions, dict):
                    history_entry["model_predictions"] = {
                        key: (
                            {
                                sub_key: sub_value
                                for sub_key, sub_value in value.items()
                                if sub_key not in {"prefix_minute", "timesteps", "artifact_mode"}
                            }
                            if isinstance(value, dict)
                            else value
                        )
                        for key, value in model_predictions.items()
                    }
                cleaned_history.append(history_entry)
            cleaned["prediction_history_preview"] = cleaned_history

        return cleaned

    def sync_featured_live_game(result: dict) -> dict:
        if not isinstance(result, dict):
            return result
        live_games = result.get("live_games")
        if not isinstance(live_games, list):
            return result

        synced_games = []
        changed = False
        for game in live_games:
            if not isinstance(game, dict):
                synced_games.append(game)
                continue
            is_same_game = (
                game.get("featured")
                and game.get("game_id") == result.get("game_id")
                and game.get("match_id") == result.get("match_id")
            )
            if not is_same_game:
                synced_games.append(game)
                continue

            updated = dict(game)
            for key in (
                "time_s",
                "time_raw",
                "time_source",
                "time_ocr_raw",
                "time_ocr_error",
                "time_sync_offset_s",
                "gold_left",
                "gold_right",
                "kills_left",
                "kills_right",
                "feed_timestamp",
                "feed_age_s",
                "status",
                "status_message",
                "game_state",
                "patch_version",
                "prediction_status",
                "prediction_reason",
                "prediction_model",
                "prediction_prefix_minute",
                "prediction_timesteps",
                "prediction_current_minute",
                "prediction_blue_win_prob",
                "prediction_red_win_prob",
                "prediction_consensus_blue_win_prob",
                "prediction_consensus_red_win_prob",
                "prediction_comparison",
                "model_predictions",
                "participants_left",
                "participants_right",
                "live_log_path",
                "live_log_skipped",
                "prediction_history_preview",
            ):
                if key in result:
                    updated[key] = result.get(key)
            synced_games.append(updated)
            changed = True

        if not changed:
            return result
        result = dict(result)
        result["live_games"] = synced_games
        return result

    def worker():
        while not stop_event.is_set():
            try:
                if source == "api":
                    result = fetch_live_scoreboard(api_client)
                    if timer_browser is not None and isinstance(result, dict):
                        should_use_live_page_match = (
                            result.get("status") in {"api_unavailable", "no_live_event"}
                            or not result.get("team_left")
                            or not result.get("team_right")
                        )
                        if should_use_live_page_match:
                            page_info = timer_browser.capture_match_info().as_dict()
                            result = merge_live_page_match_info(result, page_info)
                    is_live_game = (
                        isinstance(result, dict)
                        and result.get("status") == "in_progress"
                        and result.get("game_id") is not None
                    )
                    should_use_live_page_timer = (
                        timer_browser is not None
                        and isinstance(result, dict)
                        and result.get("status") == "page_match_only"
                    )
                    if should_use_live_page_timer:
                        time_result = timer_browser.capture_time()
                        result = apply_timer_result(
                            result,
                            time_result.time_s,
                            time_result.time_raw,
                            "lolesports_browser_ocr",
                            time_result.error,
                        )
                    elif not is_live_game:
                        result = dict(result)
                        result["time_source"] = "api"
                        result["time_ocr_raw"] = None
                        result["time_ocr_error"] = None
                    elif timer_browser is not None:
                        time_result = timer_browser.capture_time()
                        result = apply_timer_result(
                            result,
                            time_result.time_s,
                            time_result.time_raw,
                            "lolesports_browser_ocr",
                            time_result.error,
                        )
                    elif profile is not None and timer_source == "stream":
                        try:
                            timer_profile = pick_timer_profile(result)
                            scan_result = scan_stream_once(
                                timer_profile,
                                channel=channel,
                                quality=quality,
                            )
                            result = merge_ocr_scoreboard(result, scan_result)
                            result = apply_timer_result(
                                result,
                                scan_result.get("time_s"),
                                scan_result.get("time_raw"),
                                "ocr",
                            )
                        except Exception as exc:
                            result = apply_timer_result(result, None, None, "ocr", str(exc))
                    elif profile is not None and timer_source == "match-stream":
                        timer_stream = (result.get("timer_stream") or {}) if isinstance(result, dict) else {}
                        stream_candidates = result.get("stream_candidates") or []
                        ordered_streams = []
                        if timer_stream:
                            ordered_streams.append(timer_stream)
                        for candidate in stream_candidates:
                            if candidate and candidate not in ordered_streams:
                                ordered_streams.append(candidate)
                        result = dict(result)
                        result["time_stream_provider"] = timer_stream.get("provider")
                        result["time_stream_url"] = timer_stream.get("url")
                        timer_errors = []
                        for stream in ordered_streams:
                            stream_url = stream.get("url")
                            if not stream_url:
                                continue
                            result["time_stream_provider"] = stream.get("provider")
                            result["time_stream_url"] = stream_url
                            try:
                                timer_profile = pick_timer_profile(result)
                                scan_result = scan_stream_once(
                                    timer_profile,
                                    quality=quality,
                                    source_url=stream_url,
                                )
                                result = merge_ocr_scoreboard(result, scan_result)
                                if scan_result.get("time_s") is not None:
                                    result = apply_timer_result(
                                        result,
                                        scan_result.get("time_s"),
                                        scan_result.get("time_raw"),
                                        "match_stream_ocr",
                                    )
                                    if result.get("time_source") == "match_stream_ocr":
                                        break
                                    timer_errors.append(
                                        result.get("time_ocr_error")
                                        or f"{stream.get('provider') or 'unknown'} returned invalid timer"
                                    )
                                    continue
                                timer_errors.append(
                                    f"{stream.get('provider') or 'unknown'} returned no valid timer"
                                )
                            except Exception as exc:
                                timer_errors.append(f"{stream.get('provider') or 'unknown'}: {exc}")
                        if result.get("time_source") != "match_stream_ocr":
                            result = apply_timer_result(
                                result,
                                None,
                                result.get("time_ocr_raw"),
                                "match_stream_ocr",
                                (
                                    " | ".join(timer_errors)
                                    if timer_errors
                                    else "No match-specific timer stream URL available."
                                ),
                            )
                    if predictor is not None:
                        try:
                            result = enrich_result_with_predictor(result)
                            result = enrich_live_games(result)
                        except Exception as exc:
                            result = dict(result)
                            result["prediction_error"] = str(exc)
                else:
                    result = scan_stream_once(profile, channel=channel, quality=quality)
                result = append_live_log(result)
                result = sync_featured_live_game(result)
                state.set_result(result)
            except Exception as exc:
                state.set_error(str(exc))
            stop_event.wait(interval)

    thread = threading.Thread(target=worker, name="live-scan-worker", daemon=True)
    thread.start()

    @app.get("/")
    def index():
        return no_cache(Response(INDEX_HTML, mimetype="text/html"))

    @app.get("/api/state")
    def api_state():
        manual_snapshot = manual_state.snapshot()
        if manual_snapshot["active"]:
            result = strip_interface_prediction_metadata(manual_snapshot["result"])
            return no_cache(jsonify(
                {
                    "result": result,
                    "error": None,
                    "updated_at": manual_snapshot["updated_at"],
                }
            ))
        snapshot = state.snapshot()
        if isinstance(snapshot.get("result"), dict):
            snapshot = dict(snapshot)
            snapshot["result"] = strip_interface_prediction_metadata(snapshot["result"])
        return no_cache(jsonify(snapshot))

    @app.get("/api/manual-state")
    def api_manual_state_get():
        return jsonify(manual_state.snapshot())

    @app.post("/api/manual-state")
    def api_manual_state_post():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Expected a JSON object payload."}), 400
        try:
            result = prepare_manual_result(payload)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
        manual_state.set_result(result)
        return jsonify({"ok": True, "result": result})

    @app.delete("/api/manual-state")
    def api_manual_state_delete():
        manual_state.clear()
        return jsonify({"ok": True, "active": False})

    @app.post("/api/replay-file")
    def api_replay_file():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Expected a JSON object payload."}), 400
        csv_path = str(payload.get("path") or "").strip()
        if not csv_path:
            return jsonify({"error": "Missing 'path' for replay file."}), 400
        target_game_id = str(payload.get("game_id") or "").strip() or None
        try:
            replay = run_replay_file(csv_path, target_game_id=target_game_id)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"ok": True, **replay})

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--source", choices=("ocr", "api"), default="ocr")
    p.add_argument("--channel", default="riotgames")
    p.add_argument("--quality", default="best")
    p.add_argument("--profile", default=DEFAULT_PROFILE_PATH)
    p.add_argument("--interval", type=float, default=5.0)
    p.add_argument(
        "--timer-source",
        choices=("match-stream", "lolesports", "stream", "none"),
        default="match-stream",
        help="Timer OCR source used in API mode. Default: match-stream",
    )
    p.add_argument("--timer-url", default="https://lolesports.com/live")
    p.add_argument("--timer-headful", action="store_true")
    p.add_argument(
        "--live-log-dir",
        default="live_prediction_logs",
        help="Directory for append-only live prediction JSONL logs. Use empty string to disable.",
    )
    p.add_argument(
        "--replay-output-dir",
        default="replay_prediction_logs",
        help="Directory for CSV summaries created by /api/replay-file.",
    )
    args = p.parse_args()

    app = create_app(
        source=args.source,
        channel=args.channel,
        quality=args.quality,
        profile_path=args.profile,
        interval=args.interval,
        timer_source=args.timer_source,
        timer_url=args.timer_url,
        timer_headful=args.timer_headful,
        live_log_dir=args.live_log_dir or None,
        replay_output_dir=args.replay_output_dir,
    )
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
