---
version: "1.0"
author: Felix Morocutti (https://github.com/FelixM2011)  # camwatch — open-source security-cam monitor
license: MIT
hermes:
  tags: [ip-camera, systemd, onnx-yolo, telegram, person-detection]
name: camwatch
description: Operate a camwatch camera daemon (windowed person detection with Telegram alerts): config changes, --status/--once diagnostics, token/chat-id wiring, systemd control.
---

# CamWatch — operating runbook for AI agents

CamWatch is a lightweight security-cam monitor (pure Python, no venv). Inside the time windows
defined in `camwatch.json` it polls each configured IP camera once per `poll_interval_sec` for a
640×480 JPEG still, and after `confirm_frames` consecutive hits (default 3) starts local YOLO11n
ONNX **CPU** inference. On a verified person episode it sends one annotated photo — or, when
multiple frames qualified, an MP4 clip — to the configured Telegram chat once per camera per
episode. Outside all windows it is fully idle: no polling, no model usage (~45 MB RAM while
active, GPU never touched until confirmation). This keeps ordinary machine work unaffected.

## Repository layout (what you need)

- `camwatch.py` — CLI + daemon loop. Flags:
  - `--status` — show which time window is active now / next up, plus per-camera episode state. Does not load the model; safe to run anytime.
  - `--once` — exactly one manual poll pass over all cameras (ignores schedule windows). Watch stdout and the log file.
  - `--once --send` — same single pass, but additionally sends an annotated photo to Telegram for any camera that confirms a person during it. This is the end-to-end wiring check.
- `camwatch_lib.py` — shared library: config loading, time windows, camera polling (`/SnapshotJPEG?Resolution=640x480&View=Normal&Count=<random>`), token resolution, Telegram upload. Token and `tg_chat_id` are read **exactly once at import** — a running process sees no config changes until restarted.
- `detector.py` — YOLO11n ONNX person detector (letterbox 640×640) + "PERSON DETECTED" banner drawing; optional one-time CPU→GPU upgrade after a confirmed person (`maybe_gpu_once`, at most once per process).
- `camwatch.json` — deployment config: cameras, schedule, thresholds, chat id. `_rel` keys are resolved relative to this file's directory. Token is **never** in this file by design.
- `skill/SKILL.md` — this runbook. Copy the whole `skill/` folder into your agent's skill directory and it becomes usable from ordinary chat ("How is my front camera right now?").
- `systemd/camwatch.service` — per-user unit template (see README §6 for installation).

## Standard actions

### Status check (no model needed)
```bash
python3 camwatch.py --status
```
Reads the current window + next start time and per-camera episode count/cooldown. This is always
the first command to run in any conversation about this daemon.

### One-shot live test
```bash
python3 camwatch.py --once           # poll every camera once, CPU inference on hits
python3 camwatch.py --once --send    # same + Telegram upload (proof the token/chat wiring works)
```

### After ANY config or `.env` change — mandatory restart
```bash
systemctl --user restart camwatch && sleep 2 && journalctl --user -u camwatch -n 15
```
Then verify **real** delivery with `python3 camwatch.py --once --send`: the Telegram answer (or a
small test image uploaded via the lib) must succeed. The log line `[init] Telegram active: YES`
only proves *token found* — it does not prove the bot is in that exact chat (missing bot → silent
`chat_not_found`).

## Configuration keys (camwatch.json)

| Key | Meaning |
|---|---|
| `cameras[].id` / `.name` | short label + free-text name shown in logs and Telegram texts; conventionally the last IP octet. |
| `cameras[].root` | camera **base URL only**: scheme + host (+ port if not 80), no path, ideally no trailing slash. camwatch appends the snapshot query itself (one supported endpoint format: `/SnapshotJPEG?...`). Verify from your machine with a raw curl — the first bytes must be `FF D8 …` (JPEG), never an HTML login page. |
| `tg_chat_id` | numeric Telegram chat id, keep its sign; groups are negative (`-10xxxxxx`). See README §4 for how to find it via @BotFather + getUpdates. |
| `env_file` | absolute path of a dotfile holding the token line (e.g. `TELEGRAM_BOT_TOKEN="…"`) — or set that env variable directly; resolution: env var first, then file. Neither ⇒ **dry-run** mode (everything works except sending). |
| `poll_interval_sec` | per-camera poll period in seconds; values below 2 s are raised to exactly 2 s by the code floor (shipped default `2.0`). Reaction time = at most one poll + the confirm streak: with `confirm_frames=3` that is roughly one interval per frame before confirmation, so keep it small (default 2 s). |
| `confirm_frames` | consecutive hitting frames required before any inference/alert (default 3; single noise blips never fire). |
| `grace_without_person_sec` | end-of-episode grace period after the last hit while an alert is still allowed to be sent (default 90 s). |
| `conf_threshold` / `iou_threshold` | YOLO confidence (0.35) / NMS IoU (0.45). |
| `cpu_threads` | inference threads; keep low by default, so normal work is uninterrupted (GPU/extra RAM only after confirm, and only on hosts actually having a CUDA-capable Python build — CPU-only otherwise, expected behaviour there!). |
| `photo_every_sec` | minimum spacing between alerts of the same camera during one episode. |
| `video_fps_frame_gap`, `rec_dir_rel`, `model_rel` | see shipped example JSON + README §2 for meanings/defaults (MP4: source serves ~1 frame per 3 s, gap 1.7 s; model file must exist next to the config). Outside windows the daemon sleeps in short capped waits and re-checks the clock — no camera contact at all. |

Time windows `windows_by_weekday_min`: minutes since midnight per weekday, half-open `[start, end)`,
Monday = `"0"` … Sunday = `"6"`. Empty list → that day never polls. All local time of the host.

## Privacy hard rules (never break these)

- The Telegram token comes **only** from an environment variable or the `env_file` dotfile; it must appear in none of: log lines, stdout, JSON files, state files, chat replies, commit messages.
- Never paste camera IPs + chat id into public places together with tokens — deployment values are personal and stay local except this repo's code.
- The model file `yolo11n.onnx` and per-camera stills never leave the host (single Telegram destination only).

## Pitfalls

- Config is import-once: without a daemon restart, edits to camera list / thresholds / windows are silently ignored by a running process.
- Polling answers can be HTML or redirects when Referer/UA header handling fails; detection of valid stills is done on JPEG magic bytes (`FF D8`), never trust content-type headers.
- A single frame hit below `confirm_frames` must not alert — cooldown/debounce state lives in the host's `state.json`, keep it between runs.
- Numbers from CLI are often floats (ms, confidence): format with `:.0f`/`.4g`, not bare `:d`.
- When patching this codebase for a deployment, re-read the current file on disk before editing (running instances may rewrite files), and always test with a real `--once` run — `py_compile` alone proves no runtime names.
