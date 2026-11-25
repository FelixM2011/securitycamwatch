# camwatch — windowed person detection for IP cameras with Telegram alerts

**camwatch** is a lightweight security-cam monitor. Inside configured **time windows** it polls
a set of ONVIF-style IP cameras for 640×480 still snapshots, runs local YOLO11n person inference
only starting from the `confirm_frames`-th consecutive hit, and once confirmed sends an
**annotated alert photo immediately; further annotated photos follow while the person stays
visible (every `photo_every_sec`, default 30 s)** — and on episode end additionally a recorded
MP4 clip of the whole visit if one rendered validly. Outside every window it stays idle —
no polling at all — so RAM usage is near zero and the GPU is never touched until confirmation.

Built for homes / small offices: plain **system Python 3**, no virtualenv, no database, stable
low footprint (~45 MB while active).

## How it works

| Stage | What happens | Cost |
|---|---|---|
| Outside windows | main loop just sleeps in short capped waits (**at most** `max(6 × poll_interval, 27 s)` each), re-checking the clock after every one and opening episodes again at the next window start — no camera contact, no inference | ≈ none |
| In window | one `/SnapshotJPEG?Resolution=640x480&View=Normal&Count=<random>` per camera each poll (per `poll_interval_sec`, hard floor of 2 s in the code) + a local YOLO11n person inference on that snapshot, staggered by half an interval between cameras | small CPU time per poll (`cpu_threads`, default 2); no GPU involved yet |
| streak of `confirm_frames` (default 3) consecutive person hits | **episode starts**: ffmpeg recording begins for this camera + one instant annotated "PERSON DETECTED" Telegram photo; one-time attempt to upgrade the detector to CUDA happens exactly here, and from now on further photos follow at most every `photo_every_sec` while a person is visible in the current snapshot (default 30 s) | CPU; GPU only if a CUDA-capable build/driver exists and the probe succeeded |
| no hit for `grace_without_person_sec` (default 90 s), window ends, or daemon shutdown | episode closes: ffmpeg finalizes and, when the resulting MP4 is usable (> 4 kB), that clip of the whole visit is sent to Telegram with a caption (`ffmpeg` required) | — |

Anti-spam rules are enforced in the main loop and persisted in `state.json`, so restarting the
daemon never "forgets" a person inside its cooldown. Single-frame blips (noise, birds,
reflections) can **never** trigger anything — only streaks of ≥ 3 consecutive hits count.

## Repository layout

```
camwatch.py            # CLI entry point + long-running daemon loop
detector.py            # YOLO11n ONNX person detector + "PERSON DETECTED" banner drawing
camwatch_lib.py        # shared library: config, logging, time windows, camera polling, Telegram driver
skill/SKILL.md         # AI-agent runbook (English): how an agent should operate this daemon from chat
systemd/camwatch.service  # per-user systemd unit template to run camwatch as a service
camwatch.json          # annotated config TEMPLATE: one example camera entry + default thresholds (see §2); copy and edit per deployment. Tokens/chat ids are intentionally NOT in this file.
recordings/            # rendered MP4 episode stills (created at runtime — private, git-ignored)
state.json             # cooldown/debounce persistence across restarts (runtime — git-ignored)
yolo11n.onnx           # model weights (~6 MB network YOLO ONNX export). NOT in this repo on purpose: you must place or reference your own working local copy next to camwatch.py (model_rel points at it by default); there is no automatic download anywhere in the code.
```

## Getting started

### 1. Dependencies (system Python, **no venv**)

```bash
python3 -m pip install --user onnxruntime numpy "Pillow>=9"   # or distro equivalents
ffmpeg                        # only needed for MP4 episode rendering
```

camwatch itself needs nothing else: all runtime modules are the three local files in this repo.

### 2. Cameras and schedule — `camwatch.json` (annotated template)

The shipped `camwatch.json` is a ready-to-edit **annotated template**: every key carries an inline
`_comment` or `_note_*` sibling explaining its meaning (JSON has no comments, so these extra keys
serve that purpose — the code ignores any key it does not know). Copy it to your own file name and edit:

```bash
cp camwatch.json my_cameras.json && $EDITOR my_cameras.json
# then point the daemon at it if you renamed it (systemd unit Environment=CAMWATCH_CONFIG=/path/to/my_cameras.json)
```

**Camera list — one object per device:**

| Key | Meaning | Example / notes |
|---|---|---|
| `id` | short label used in logs and Telegram texts (`[cam 01]`) | any string, commonly the last IP octet: `"01"` |
| `name` | free-text name shown next to `id` | e.g. `"Driveway"`, `"Front door"` … |
| `root` | **camera base URL only** — scheme + host (+ port if not 80). No path, no query string, preferably no trailing slash: camwatch appends `/SnapshotJPEG?Resolution=640x480&View=Normal&Count=<random>` itself — same URL shape as your `root`, plus that suffix. Add one object per camera. | `"http://192.168.X.Y"` (your network) |

`root` quick test from your machine (must return **live JPEG bytes**, not HTML):

```bash
curl -s "http://192.168.X.Y/SnapshotJPEG?Resolution=640x480&View=Normal&Count=" \
  | xxd | head -1        # first line must start with FF D8 … (a JPEG), not 'HT'/'<' of an HTML page
```

If the browser test returns a login page or redirect instead, that camera needs its own protocol
adapter and has to sit behind one before camwatch can use it — there is exactly **one** supported
snapshot endpoint format (`/SnapshotJPEG?…`), so no multi-vendor guessing happens at runtime.

**Time windows (`windows_by_weekday_min`)**: minutes since midnight per weekday; each entry is a
half-open `[start, end)` pair (the stop moment itself counts as *outside*). Weekday keys are
**Monday = "0" … Sunday = "6"** (Python `weekday()` numbering). An empty list for a day means that
day is never polled.

```json
"windows_by_weekday_min": {
    "0": [[480, 780], [1380, 1440]],   // Mon (example): 08:00–13:00 and 23:00–midnight
    ...                                // one entry per weekday; edit to your own schedule
}
```

**All other keys** (defaults from the shipped template): `model_rel` (`"yolo11n.onnx"` next to camwatch.py; must already exist locally — no automatic download exists anywhere), `conf_threshold` = 0.35 (min YOLO confidence per single hit),
`iou_threshold` = 0.45, `confirm_frames` = 3, `grace_without_person_sec` = 90,
`photo_every_sec` = 30 (minimum spacing between consecutive verified alerts for one camera),
`poll_interval_sec` = base cadence per-camera (shipped at `2.0`; the code hard-clamps any value below 2 s up to exactly 2 s), `cpu_threads` = 2 (kept low on purpose — CPU-only inference so your GPU stays free);
`video_fps_frame_gap` = 1.7 s, `tg_chat_id`, `tg_token_key` (default `"TELEGRAM_BOT_TOKEN"`),
`env_file`, `rec_dir_rel` (`"recordings"`). The complete shipped `camwatch.json` remains the
authoritative per-key reference — edit a *copy*, change cameras/chat/windows per deployment.

### 3. Telegram token — keep it out of JSON entirely

Resolution order at startup (first hit wins): environment variable named by `tg_token_key` → then
the file pointed to by `env_file`. Neither present ⇒ explicit **dry-run** start message, and every
feature works except actual sending:

```bash
# option A — environment variable on the host running it
export TELEGRAM_BOT_TOKEN=1234567890:AABBCC…        # before starting / in the systemd unit's Environment= line

# option B — secrets dotfile, exact format below (value may or may not be quoted)
TELEGRAM_BOT_TOKEN="1234567890:AABBCC…"             # in an absolute path of your choosing → set to env_file in camwatch.json
```

Never put the token into `camwatch.json` itself — it is not read from there by design. The daemon
masks the token in logs/state even under verbose debugging (`CAMWATCH_DEBUG=on` gives per-frame
details).

### 4. Finding your chat id for `tg_chat_id`

1. [@BotFather](https://t.me/BotFather) → create a bot, note its token (step 3).
2. Add the bot **as member** to the personal or group chat that should receive alerts.
3. Send any message *to* that chat from yourself, then once on your machine:

```bash
curl -s "https://api.telegram.org/bot<TOKEN>/getUpdates" | grep '"chat"'
# → look for "id": <number>; groups/supergroups are negative (-100xxxxxxxxxx) — keep the sign; that number is tg_chat_id
```

`tg_chat_id` missing/invalid ⇒ same dry-run behaviour as with a missing token (everything else keeps working).

### 5. One-shot tests before going live

From the repo root, plain user:

```bash
python3 camwatch.py --status        # which window is active right now, next one if none, per-camera episode state — no model loading needed
python3 camwatch.py --once          # exactly ONE manual poll pass over all cameras this instant (ignores scheduled windows by design) — watch stdout + log file
python3 camwatch.py --once --send   # same single pass, but additionally sends the annotated still to Telegram for any camera that confirmed a person now — end-to-end wiring check before trusting it with a service manager. (--send is only meaningful together with --once.)
```

### 6. Run under systemd (per-user unit supplied)

`systemd/camwatch.service` targets `default.target` in the **user** session — no root needed:
set `WorkingDirectory=<repo checkout>`, point `ExecStart=` at your python3 and this checkout's
`camwatch.py`; add any extra environment (`CAMWATCH_DEBUG=on`, `PATH=/home/you/…`) under
`[Service] Environment=`. Then on the target host:

```bash
mkdir -p ~/.config/systemd/user && cp systemd/camwatch.service ~/.config/systemd/user/ && $EDITOR ~/.config/systemd/user/camwatch.service
systemctl --user daemon-reload && systemctl --user enable camwatch && systemctl --user start camwatch
journalctl --user -u camwatch          # add -f to tail forever during bring-up
```

`Restart=on-failure` + short `RestartSec=` = systemd auto-respawns on unhandled exceptions; no custom watchdog code required. Editing any config needs exactly one `systemctl --user restart camwatch`.

### 7. Optional: AI-agent operation from a chat bot (English runbook shipped)

Copy the whole `skill/` folder next to wherever your assistant keeps its agent skills / runbooks
(e.g. this project's own profile uses it under `~/.hermes/profiles/<profile>/tools/camwatch/`).
The English `SKILL.md` tells any agent exactly which CLI flags do what, config key meanings per
deployment type, and troubleshooting shortcuts — after that you simply ask in natural chat, e.g.
*"how are my cameras right now?"*, and the loaded skill makes the model call `--status` /
one-oneshot combinations instead of trying anything live from the prompt itself.

## Design notes (why it looks this way on purpose)

- **CPU-first by default:** ONNX-Runtime CPU inference with a deliberately low thread count (`cpu_threads`, default 2); GPU/extra RAM only kick in *after* a confirmed person streak — never preemptively. On hosts with a CUDA-capable Python build the detector upgrades to GPU exactly once, for the rest of that process; on CPU-only machines it simply stays on CPU.
- **Strictly serial per-camera loop, no threads at all:** keeps RAM fully predictable (~45 MB while active) even without live stream decoding.
- **Persistent debounces/cooldowns written atomically** to `state.json` (temp file + rename): a crash or power loss never leaves it half-written and never double-alerts the same person.
- **Uniform log prefixes** `[cam 38] HH:MM:SS …`: every line is greppable by camera and time in one terminal command; no per-camera log files.
- **RAM-only inside windows:** outside them nothing runs at all (no polling). Snapshot requests are deliberately small (`_MIN_SNAP_BYTES` guard skips anything that looks like an HTML page before the JPEG check even happens).
- **systemd is the only auto-respawn mechanism** (`Restart=on-failure`, short `RestartSec=`); no custom watchdog in Python — config edits require an explicit restart, exactly as documented above.

## Troubleshooting (verified symptom ↔ root cause — exact log strings from the current code)

| Symptom you will see | Likely real cause | Fix / verification |
|---|---|---|
| At boot: `[init] Telegram INACTIVE (token=missing; ChatID=None -> dry-run only)` or `[init] env_file not found at … -- telegram mode DISABLED (dry-run)` | No token/chat id visible to the daemon process. | Set an absolute `env_file` path (§3) and/or export the variable, restart the service, then prove delivery end-to-end with `python3 camwatch.py --once --send`. Absence of these INACTIVE lines on startup is the positive check (no line = token + chat id resolved). |
| Startup fails with `[init] model not loadable (…) — daemon shutting down`, or the streak never reaches a line like `[cam 10] person confirmed after 3 frames -- episode starts.` for a real person | No working ONNX under `model_rel` / inference too slow. | Keep the locally proven local YOLO file in place (§2: no automatic download exists), verify `python3 camwatch.py --once` completes, and watch one full episode in the log (poll → confirmed → photo/episode start). |
| A camera whose browser view shows a login or an HTML page but camwatch logs `no JPEG answer (…) bytes` — i.e. the daemon keeps receiving something that is not a JPEG snapshot | The root has no trailing-slash-expected form: poll builds `<root>/SnapshotJPEG?...`; the extra path element breaks the URL so a portal/HTML response comes back instead of JPEG bytes. | Ensure `root` ends exactly at the host (no trailing `/`) for this camera class, check the Referer handling in `camwatch_lib.poll_cam`, and validate on the same host with a raw curl before restarting the service. |
| Telegram line logs `[telegram] upload failed …` despite token + chat configured correctly | Wrong chat id (bot never added to that exact group), or stale id/sign mismatch inside groups/supergroups. | Re-derive `tg_chat_id` via @BotFather + getUpdates on a *new* message sent into the target chat (§4), restart, then run `--once --send`. Do NOT paste tokens anywhere public while doing this (mask in logs). |
| Alert appears hours after visible person / no alert at all for sustained visits | Long streak requirement vs. snapshot cadence; grace/cooldown state from previous episode still active (`state.json`). | Watch `[cam N] … confirmed` line: it only fires once `confirm_frames` consecutive hits pile up without a miss between; lower `confirm_frames` or tune the camera to give cleaner snapshots, then confirm with another live episode before restarting. |
*MIT License · repository by FelixM2011 (Felix Morocutti) — built from a real ONVIF-style residential-office deployment; duplicate `camwatch.json` as your own config per fork (§2), edit camera list + chat id, and keep tokens in your environment or dotfile.*
