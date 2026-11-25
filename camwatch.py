#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""camwatch -- network-camera person detection as a systemd user daemon.

Flow (only inside the time windows defined in camwatch.json):
  * Each camera serves ~one still every 3 s (no login required); we poll per
    camera on a configured beat. The second camera starts offset (staggered
    phases) so both never ask at exactly the same moment.
  * Detection = YOLO11n ONNX, intentionally CPU-only at start (the GPU stays
    free for normal requests). Only once a person is CONFIRMED
    (confirm_frames consecutive hits), exactly one GPU switch probe runs; if it
    fails, the process definitively continues on CPU.
  * Confirmed person => episode:
      - ffmpeg recording starts; EVERY received photo flows into the MP4
      - immediate annotated photo + Telegram "person detected" (banner in image)
      - while the person is visible: further photos every photo_every_sec seconds
  * grace_without_person_sec without a person => clean close of the recording,
    finished MP4 sent to Telegram.
  * Outside the time windows: no polling (RAM-first); open episodes are closed
    cleanly at the window edge.

Commands:
  camwatch.py              continuous surveillance (systemd)
  --status                 show current state + live windows
  --once [--send]          one manual poll run over all cameras (windows ignored);
                           with --send additionally send an annotated photo to
                           Telegram when a person is detected.
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # systemd starts with CWD=/ -- absolute paths are mandatory.
sys.path.insert(0, BASE_DIR)

import camwatch_lib as cl
from detector import Detector, annotate_frame

CFG     = cl.CFG
RECDIR  = cl.RECDIR

STOP           = False                          # clean shutdown on signal (daemon mode)
DET          = None                            # exactly one detector instance per process
RUN: dict     = {}                             # cam_id -> {"proc": Popen|None, "out": str} (not serializable! -> NOT part of the state file)
STATE: dict   = {"updated": "", "inside_windows": False, "cams": {}}

_POLL_INTERVAL   = max(float(CFG.get("poll_interval_sec", 3.0)), 2.0)    # hard floor: 8 s reaction felt too slow live; never poll faster than 2 s per camera.
_CONFIRM_FRAMES  = max(int(CFG.get("confirm_frames", 3)), 1)
_GRACE_SEC       = float(CFG.get("grace_without_person_sec", 90))
_PHOTO_EVERY_SEC = float(CFG.get("photo_every_sec", 30))
_FRAME_GAP      = max(float(CFG.get("video_fps_frame_gap", 3.0)), 1.5)   # the source serves ~1 frame/3 s -- video rate matches that grid.
_STAGGER        = _POLL_INTERVAL / 2.0                 # second camera starts half an interval offset
_MIN_FREE_MB     = float(CFG.get("min_free_mb", 2048))

_CAM_ID_FAILS: dict       = {}    # camera id -> running failure counter (throttled warning only)
_LAST_STATE_SAVE   = 0.0


# ------------------------------------------------------------------ Basics ---------------------------------------------------------

def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _ts(s):                      # ISO time string -> datetime | None (for comparing state fields)
    try:
        return datetime.strptime(str(s), "%Y-%m-%d %H:%M:%S") if s else None
    except Exception:
        return None


def _sleep(wait_sec: float) -> None:            # STOP-interruptible sleep (~1 s stepping)
    end = time.monotonic() + max(0.0, wait_sec)
    while not STOP and time.monotonic() < end:
        step = min(1.0, end - time.monotonic())
        if step > 0:
            time.sleep(step)


def _new_cam_state(cam: dict) -> dict:          # serializable state field set per camera
    return {
        "id": str(cam["id"]),
        "name": cam.get("name", ""),
        "streak": 0,
        "active": False,
        "episode_started_at": None,
        "last_person_seen": None,
        "next_photo_at": None,
        "frames_fed": 0,
        "last_episode": None,                   # {"started","ended","reason","video"} of the last episode
    }


def _save_state(force: bool = False) -> None:   # throttled (~25 s), otherwise immediate on events
    global _LAST_STATE_SAVE
    t = time.monotonic()
    if not force and (t - _LAST_STATE_SAVE) < 25.0:
        return
    _LAST_STATE_SAVE = t
    STATE["updated"] = _iso(datetime.now())
    cl.write_state(STATE)


def ensure_dirs() -> None:                      # make sure recordings/ + samples/ exist (idempotent)
    for d in (RECDIR, f"{BASE_DIR}/samples"):
        try:
            os.makedirs(d, exist_ok=True)
        except Exception as e:
            cl.dbg(f"[dirs] makedirs({d}) failed ({type(e).__name__}: {str(e)[:80]}) -- continuing without this path.")


# ------------------------------------------------------------------ ffmpeg recording ---------------------------------------------------

def _start_recorder(cam_id: str):               # -> (Popen|None, mp4_path); failure => episode continues without video
    gap = f"1/{_FRAME_GAP:.4f}"                 # e.g. 0.588 frames/s -- covers the source's ~3 s grid
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"{RECDIR}/cam{cam_id}_{ts}.mp4"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "image2pipe", "-framerate", gap, "-i", "-",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p", "-r", gap,
        "-c:v", "libopenh264", "-b:v", "512k", "-preset", "veryfast",
        "-movflags", "+faststart", "-t", str(int(7200)), out,            # -t caps at 2 h (safety limit against endless files)
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        cl.log(f"[rec {cam_id}] ffmpeg start failed ({type(e).__name__}: {str(e)[:120]}) -- episode continues without video.")
        return (None, out)
    cl.dbg(f"[rec {cam_id}] started: {out} ({gap} frames/s)")
    return (proc, out)


def _feed(rec, frame: bytes) -> bool:           # one JPEG frame into the ffmpeg pipe; False => pipeline is dead
    if rec is None or getattr(rec.stdin, "closed", True):
        return False
    try:
        rec.stdin.write(frame)
        rec.stdin.flush()
        return True
    except (BrokenPipeError, OSError):
        return False


def _finish_recorder(proc, out_path: str):      # clean shutdown; returns the path ONLY when the video file is usable
    if proc is None:
        return None
    try:
        if proc.poll() is None and not getattr(proc.stdin, "closed", False):
            proc.stdin.close()                            # EOF => ffmpeg finalizes the moov atom cleanly
    except OSError:
        pass
    try:
        proc.wait(timeout=12)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass
    if os.path.isfile(out_path) and os.path.getsize(out_path) > 4096:   # <4 kB = broken/empty video -> do NOT pass it off as a video
        return out_path
    cl.dbg(f"[rec] dropping file (too small): {out_path}")
    try:
        if os.path.isfile(out_path):
            os.remove(out_path)
    except OSError:
        pass
    return None


# ------------------------------------------------------------------ Telegram helpers --------------------------------------------------------

def _send_photo(jpeg_bytes: bytes, cid: str) -> bool:
    """Save the annotated photo + send it to Telegram; True if sending was ok (details in log)."""
    fn = f"{RECDIR}/notify_{cid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    try:
        with open(fn, "wb") as fh:
            fh.write(jpeg_bytes)
        status, info = cl.tg_send_file(fn, None)          # caption stays empty -- the banner in the image carries the message itself.
        if status != "ok":                                 # failures are NOW visible (before: debug level only -> invisible!)
            cl.log(f"[cam {cid}] photo send failed ({os.path.basename(fn)}): {str(info)[:120]}")
            return False
        cl.log(f"[cam {cid}] notify photo sent OK ({os.path.basename(fn)}).")   # success log line = audit trail (was silent before!)
        return True
    except Exception as e:                                # file/telegram errors are NOT fatal -- the episode keeps running
        cl.log(f"[cam {cid}] photo send failed ({type(e).__name__}: {str(e)[:80]}) -- episode continues.")
        return False


def _video_cap(cid: str, dur_min: int) -> str:            # caption for the closing video (with camera name from config)
    cam  = next((c for c in cl.CAMERAS if str(c["id"]) == cid), {})
    nm   = cam.get("name") or f"Cam {cid}"
    return f"{nm}: person episode, duration {dur_min:d} min."


# ------------------------------------------------------------------ Episode logic ----------------------------------------------------------

def _stop_episode(cid: str, reason: str) -> None:
    cid, st  = str(cid), (STATE["cams"].setdefault(str(cid), {"episode_started_at": None}))
    info     = RUN.pop(str(cid), {})
    began    = _ts(st.get("episode_started_at")) if isinstance(st.get("episode_started_at"), str) else None
    ended_dt = datetime.now()
    dur_min  = int((ended_dt - began).total_seconds() // 60) if began else 0
    video    = _finish_recorder(info.get("proc"), info.get("out") or "")
    cap      = f"{_video_cap(cid, dur_min)} ({reason})"
    if video:
        status, info_txt = cl.tg_send_file(video, cap)
        if status != "ok":
            cl.log(f"[cam {cid}] video send failed -- file remains at {video} (reasons in log).")
    else:
        cl.dbg(f"[rec {cid}] episode ended without a usable video ({cap})")
    st["active"]     = False
    st["last_episode"] = {"started": st.get("episode_started_at"), "ended": _iso(ended_dt),
                          "reason": reason, "video": video,
                          "frames_fed": int(st.get("frames_fed", 0))}
    _save_state(force=True)


def process_cam(cam: dict, nowdt: datetime) -> None:     # ONE poll + processing for a single camera
    cid   = str(cam["id"])
    st    = STATE["cams"].setdefault(cid, _new_cam_state(cam))
    st.setdefault("name", cam.get("name", ""))

    frame = cl.poll_cam(cam)                              # None on timeout/network error or non-JPEG reply

    if not (frame and frame[:2] == b"\xff\xd8"):          # errors count for neither streak nor against grace
        f           = _CAM_ID_FAILS.get(cid, 0) + 1
        _CAM_ID_FAILS[cid] = f
        if f in (5, 20, 60):                              # throttled warning on ongoing image loss
            cl.log(f"[cam {cid}] {f}x polls without a valid JPEG -- episode/streak logic stays untouched.")
        return                                           # -> errors count for neither streak nor against grace

    _CAM_ID_FAILS[cid] = 0

    try:                                                 # {'w','h','ms','persons':[{x1,y1,x2,y2,conf}], 'labels':[(name,conf),...]}
        res     = DET.detect(frame)                      # model/inference errors must NOT stop the round
        persons = res.get("persons") or []
    except Exception as e:
        cl.log(f"[cam {cid}] inference failed ({type(e).__name__}: {str(e)[:100]}) -- this poll is skipped.")
        return

    active_info = RUN.get(cid)
    is_active   = bool(active_info and (active_info.get("proc") is not None or active_info.get("out")))
    confirmed   = int(st.get("streak", 0)) >= _CONFIRM_FRAMES

    if persons:
        st["streak"]           = int(st.get("streak", 0)) + 1
        st["last_person_seen"] = _iso(nowdt)

        # === just confirmed -> start episode (first activation in this person phase) ===
        if not is_active:
            proc, out = _start_recorder(cid)
            RUN[cid] = {"proc": proc, "out": out}
            st.update({
                "active": True,
                "episode_started_at": _iso(nowdt),
                "frames_fed": 0,
                "next_photo_at": None,                    # -> the path below sends the FIRST photo immediately
            })
            is_active = True                              # valid for the feed/send logic below
            cl.log(f"[cam {cid}] ({st.get('name')}) person confirmed after {int(st['streak'])} frames -- episode starts.")
            DET.maybe_gpu_once()                          # exactly ONE GPU attempt per process; failure => CPU forever

        # === every received photo flows into the video (as long as the pipeline is alive) ===
        if is_active and RUN.get(cid, {}).get("proc") is not None:
            if _feed(RUN[cid]["proc"], frame):
                st["frames_fed"] = int(st.get("frames_fed", 0)) + 1

        # === first notification photo IMMEDIATELY on confirmation; then every photo_every_sec -- only while the person is visible! ===
        np_at = _ts(st.get("next_photo_at"))
        do_send = (np_at is None) or (nowdt >= np_at)    # None -> this is the FIRST photo -> send
        if st.get("active") and persons and do_send:
            _send_photo(annotate_frame(frame, res), cid)     # green boxes + red banner "PERSON DETECTED <time>" from the real result
            st["next_photo_at"] = _iso(nowdt + timedelta(seconds=_PHOTO_EVERY_SEC))

        if is_active and RUN.get(cid, {}).get("proc") is None:
            cl.dbg(f"[rec {cid}] episode active but video NOT running (ffmpeg dead) -- photos/logic continue.")

    else:                                                # --- no person in this frame ---
        st["streak"] = 0                                 # streak reset; grace keeps counting from last_person_seen below

    # === grace check: only for active episodes, person gone for longer than _GRACE_SEC ===
    if st.get("active"):
        lp = _ts(st.get("last_person_seen"))
        if persons == [] and (lp is None or (nowdt - lp).total_seconds() >= _GRACE_SEC):
            idle  = int((nowdt - lp).total_seconds()) if lp else 0
            nm    = st.get("name") or cid
            cl.log(f"[cam {cid}] ({nm}) episode ends -- person out of view for ~{idle} s now.")
            _stop_episode(cid, "person gone" + (f" after ~{idle} s" if lp else ""))


# ------------------------------------------------------------------ Main program ------------------------------------------------------------

def _status() -> int:                                    # --status : compact live status against the current clock
    st           = cl.read_state() or {}
    cams         = st.get("cams") or {}
    active, wins = cl.in_window(datetime.now())
    if active and isinstance(wins, list) and wins:
        fmt = ", ".join(cl.fmt_range(w) for w in wins)
    else:
        s  = cl.secs_until_next_start(datetime.now())
        nxt = datetime.fromtimestamp(time.time() + s)
        fmt = f"none active -- next window from {nxt:%H:%M} ({cl.WEEKDAYS_FULL[nxt.weekday()]})"
    print(f"camwatch status (updated {st.get('updated') or '--'}); windows today: {'ACTIVE ' + fmt if active else fmt}")
    for cid, cs in cams.items():
        ep   = cs.get("last_episode") or {}
        act  = ("ACTIVE (episode since " + str(cs.get("episode_started_at"))[:16] + ")") if cs.get("active") else \
               (f"last episode: {str(ep.get('ended') or '--')[:16]} ({ep.get('reason','?')}), {int(ep.get('frames_fed',0))} frames" if ep else "no episode yet")
        print(f"[cam {cid}] ({(cs or {}).get('name','')}) streak={cs.get('streak',0)} | {act}")
    return 0


def _once(send: bool) -> int:                            # --once : one manual poll run over all cameras (windows ignored!)
    global DET
    ensure_dirs()
    try:
        DET = Detector(model_path=str(CFG["model"]),
                       conf_threshold=float(CFG.get("conf_threshold", 0.35)),
                       iou_threshold=float(CFG.get("iou_threshold", 0.45)))
    except Exception as e:
        print(f"[camwatch] model not loadable ({e!r}) -- --once aborted.", file=sys.stderr)
        return 2

    nowdt, rc = datetime.now(), True
    for cam in (cl.CAMERAS or []):
        cid   = str(cam["id"])
        frame = cl.poll_cam(cam)
        if not (frame and frame[:2] == b"\xff\xd8"):
            print(f"[cam {cid}] ({cam.get('name','')}) NO JPEG received (poll failed).")
            rc = False                                   # camera unreachable -> overall result NOT successful
            continue
        try:
            res     = DET.detect(frame)
        except Exception as e:
            print(f"[cam {cid}] inference failed ({e!r}) -- skipped.")
            rc = False
            continue
        persons   = res.get("persons") or []
        conf_txt  = f"{max(p['conf'] for p in persons):.2f}" if persons else "--"
        print(f"[cam {cid}] ({cam.get('name','')}) {len(persons)} person(s) (max conf {conf_txt}) -- inference {float(res.get('ms', -1)):.0f} ms.")
        if persons and send:                            # --send : additionally send an annotated photo to Telegram (windows irrelevant!)
            fn = f"{RECDIR}/once_{cid}_{nowdt:%Y%m%d_%H%M%S}.jpg"
            with open(fn, "wb") as fh:
                fh.write(annotate_frame(frame, res))
            status, info_txt = cl.tg_send_file(fn, None)     # the banner in the image carries the message -- caption empty.
            print(f"[cam {cid}] telegram send: {(status == 'ok') and 'OK' or str(info_txt)[:120]}")
    return (0 if rc else 1)


def _sweep_disk() -> None:                               # disk space below threshold => delete oldest recordings until there is room again
    try:
        free_mb = cl.disk_free_mb(RECDIR)               # None => unavailable -> treat as "infinite" (no crash!)
    except Exception:
        return
    if not free_mb or float(free_mb) >= _MIN_FREE_MB:   # enough space -> do nothing
        return
    files = []                                          # collect all candidate files from the recordings/ folder
    for f in os.listdir(RECDIR):                        # both episode classes (videos + notification photos) qualify as candidates!
        if re.fullmatch(r"cam\d+_\d{8}_\d{6}\.mp4", f) or (f.startswith("notify_") and f.endswith(".jpg")):
            full = os.path.join(RECDIR, f)
            try:
                files.append((float(os.path.getmtime(full)), full))   # mtime as comparison value (oldest first)
            except OSError:                              # deleted in the meantime? -> just skip it.
                continue
    files.sort()                                        # oldest first => exactly the "oldest" candidates are at the front!
    while float(free_mb or 0.0) < _MIN_FREE_MB and files:   # keep deleting until there is room again!
        _, fpath = files.pop(0)                          # remove the oldest one first; on error -> log + break out of the loop!
        try:
            os.unlink(fpath)                            # a single stubborn file must NOT crash the daemon!
            cl.log(f"[disk] deleted {os.path.basename(fpath)} to free space ({float(free_mb or 0.0):.0f} MB before).")   # only filename + before-value logged (no token content!).
            nf = cl.disk_free_mb(RECDIR)               # re-check after deleting; None => treat as unchanged and break out of the loop!
            free_mb = None if nf is None else float(nf)
            if free_mb is None: break                   # no longer measurable? -> abort (no crash!)
        except OSError as e_:                          # file gone/in use? -> abort the cleanup loop (the rest gets re-checked later!)
            cl.dbg(f"[disk] unlink failed ({e_!r})")
            break


def _daemon() -> int:                                   # continuous surveillance -- the actual camwatch logic
    global DET, STOP
    ensure_dirs()

    try:                                                 # load model (error => systemd reports the restart; clear in log!)
        DET = Detector(model_path=str(CFG["model"]),
                       conf_threshold=float(CFG.get("conf_threshold", 0.35)),
                       iou_threshold=float(CFG.get("iou_threshold", 0.45)))
    except Exception as e:
        cl.log(f"[init] model not loadable ({type(e).__name__}: {str(e)[:160]}) -- daemon shutting down (systemd shows the restart attempt).")
        return 2

    cl.log(f"[init] Telegram active: {'YES' if (cl.TG_OK) else 'NO -> dry-run (recordings/logs keep running, photos/videos are NOT sent).'}")   # named clearly exactly once -- no spam per send-interaction!

    def _sig(_signum, _frame):
        global STOP
        if not STOP:                                    # report exactly ONCE per signal burst (repeated INT presses stay quiet)
            cl.log("[signal] shutdown requested -- finishing the round cleanly...")   # second argument is a placeholder frame (no extra data needed) => single message.
        STOP = True

    try:                                                # signal setup must not kill the daemon prematurely!
        signal.signal(signal.SIGINT,  _sig)
        signal.signal(signal.SIGTERM, _sig)
    except Exception as e:                              # setup failed? Not fatal -- shutdown stays manually controllable.
        cl.log(f"[init] signal setup failed ({type(e).__name__}: {str(e)[:80]}) -- continuing without clean signal handling.")

    cams  = cl.CAMERAS or []
    for c in cams:                                      # create all camera states fresh (RESET on daemon restart)
        STATE["cams"][str(c["id"])] = _new_cam_state(c)

    t0            = time.monotonic()             # daemon start timestamp (kept for parity with the original script).
    next_due      = {}                        # cam -> next due poll timestamp (staggered phases across cameras)

    cl.log(f"[init] daemon starting with {len(cams)} camera(s); model={os.path.basename(str(CFG['model']))}; "
           f"poll={_POLL_INTERVAL}s, GPU activates ONLY after a confirmed person (otherwise permanent CPU).")

    last_state_save = 0.0                   # for throttled state saving + disk sweep below (local clock, like the original).
    prev_active     = None                  # last observed window status (to detect edge crossings!)

    while not STOP:                               # ===================== main daemon loop ====================================
        nowdt   = datetime.now()
        actives, wins_listed = cl.in_window(nowdt)      # an active time window? (list of the windows running for this weekday)

        STATE["inside_windows"] = bool(actives)         # state field: "currently inside a surveillance window"? yes/no.

        # ------------------- EDGE CROSSING: WINDOW -> OUTSIDE => close open episodes cleanly now ----------------------
        if prev_active is not None and prev_active != bool(actives):            # status just changed!
            if not actives:                                        # just stepped OUT? => end all still-running episodes (no half files left behind!).
                for cid in list(RUN.keys()):                        # explicitly close every still-open episode right here now:
                    nm = (STATE["cams"].get(str(cid)) or {}).get("name") or str(cid)
                    cl.log(f"[cam {cid}] ({nm}) window end reached -- ending the episode now.")   # definitive reason + clear name in log!
                    try:                                           # best effort: a single closing failure must NOT crash the daemon (it can keep running)!
                        _stop_episode(str(cid), f"window end - {nm}")   # the episode ends officially here with video file + telegram send.
                    except Exception as e_x:                            # defensive capture -- continued surveillance stays the priority!
                        cl.dbg(f"[cam {cid}] window-edge close failed: {e_x!r}")   # log level only (no crash break!).
            prev_active = bool(actives)                          # remember the new active status for the next edge comparison.

        # ===================== INSIDE WINDOW: poll/processing per camera / episode control ================================================
        if actives and not STOP:                     # === only ACTIVE inside the window! == OUTSIDE = no polling, GPU/CPU stay free!
            nowt = time.monotonic()                  # base time for "which camera is due next?"
            for idx, cam in enumerate(cams):         # staggered phase: second camera starts half an interval later (avoids simultaneous loading!)
                cid  = str(cam["id"])
                nd   = next_due.get(cid) or (nowt + idx * _STAGGER)        # first time: offset start. Then poll every `_POLL_INTERVAL` s per camera!
                if nowt >= nd and not STOP:                     # only when this camera is "due" -> run its poll+processing round!
                    try:                                      # an error on one single camera must NOT block the others!
                        process_cam(cam, nowdt)                 # -> poll + detect + streak/active/photo logic all together!
                    except Exception as e__:                  # defensive capture -- surveillance stays the priority.
                        cl.dbg(f"[cam {cid}] unexpected error ({type(e_).__name__}: {str(e_)[:80]})")   # log only (no crash from a single poll!).
                    next_due[cid] = time.monotonic() + _POLL_INTERVAL      # schedule the next run of this camera.

            nowt2     = time.monotonic()                        # processing timestamp for the following "when to continue?" computation.
            due_vals  = [v for v in next_due.values() if not STOP]       # only still-open cameras count (nothing more to compute on stop)!
            sleep_t   = max(0.0, min(due_vals, default=nowt2 + _POLL_INTERVAL) - nowt2)      # determine the earliest due camera.
            sleep_t   = max(0.05, min(sleep_t, _POLL_INTERVAL))   # never negative, never longer than the base interval (starvation-free).
            if not STOP and sleep_t > 0:
                _sleep(sleep_t)                                # interruptible sleep (STOP signal reacts in < ~1 s!).

        # ================ OUTSIDE WINDOW: rest until next start (no RAM poll pressure!) ===========================================
        if not actives and not STOP:                            # currently NOT inside a time window? => just wait for the next window!
            wait_s = cl.secs_until_next_start(nowdt)            # seconds until the next defined start point from windows_by_weekday_min.
            _sleep(min(max(wait_s, 1.0), max(_POLL_INTERVAL * 6, 27)))   # never longer than the capped waiting sleep (so STOP interruption possible in < ~30 s!).

        # === regular state saving + disk vigilance (only when videos are actually running) ================
        if not STOP:
            if time.monotonic() - last_state_save >= 25.0:     # roughly every 25 s (free, when nothing happens!) -> throttled write.
                _save_state(force=True)                        # force=True to skip the throttle window and ACTUALLY save now.
                last_state_save = time.monotonic()             # keep the timestamp up to date for the next comparison.
            if RUN:                                            # disk sweep ONLY when at least one active episode runs (video files grow!)!
                _sweep_disk()                                  # oldest recordings out until min_free_mb of space is back (defends the RAM-first guard!).

    # ------------------------------------------------------------------ clean shutdown ----------------------------------------------------------
    closed = []                                            # episodes closed NOW (at daemon exit) -- for the final report.
    for cid in list(RUN.keys()):                                 # still-open episodes -> close correctly now + best-effort video send!
        nm  = f"Cam {cid} ({(STATE['cams'].get(cid) or {}).get('name') or ''})".strip("() ")     # tidy name for the log + caption.
        st0 = STATE["cams"].get(cid) or {}                     # remember start time BEFORE closing (duration computation).
        began0 = _ts(st0.get("episode_started_at")) if isinstance(st0.get("episode_started_at"), str) else None
        cl.log(f"[shutdown] still-open episode: {nm}")         # clearly which camera is affected before the process exits.
        try:                                                   # best-effort close; a failure must NOT abort the finish block!
            _stop_episode(cid, f"{nm}: daemon shutdown")   # closes the episode + sends the finished video (if >= 4 kB).
        except Exception as e_e:                                # defensive capture -- shutdown attempts are allowed to fail.
            cl.dbg(f"[shutdown] episode close failed: {e_e!r}")   # debug trace only; no crash from a single error!
            continue
        le   = (STATE["cams"].get(cid) or {}).get("last_episode") or {}   # _stop_episode just wrote start/ended here.
        endd = _ts(le.get("ended")) if isinstance(le.get("ended"), str) else None
        dur_s = int((endd - began0).total_seconds()) if (began0 and endd) else None
        closed.append((nm, dur_s))                            # name + real duration in seconds for the final log line.
    _save_state(force=True)                                  # last state snapshot (for --status after exit).
    part = ("; " + ", ".join(f"{n} ({(d or 0)/60.0:.1f} min)" if d is not None else f"{n}" for n, d in closed)) if closed else ""
    cl.log(f"camwatch exited{part}, state saved.")     # clean finish with real episode durations (no unrealistic total number anymore).
    return 0


def main() -> int:                                            # entry point (parse CLI arguments & route!)
    ap = argparse.ArgumentParser(description="camwatch daemon / camera person detection")
    ap.add_argument("--status", action="store_true", help="show current state + live windows")       # compact status output.
    ap.add_argument("--once",   action="store_true", help="one manual poll run over all cameras (windows ignored)")  # single test poll (useful for debugging!)
    ap.add_argument("--send",   action="store_true", help="only with --once: additionally send an annotated photo to Telegram")      # only meaningful in once mode; ignored otherwise.
    args = ap.parse_args()

    if args.status:                                            # status request? -> print the status directly (no model loading needed!) and finish.
        return _status()
    if args.once:                                              # manual single poll wanted? -> call the --once routine (with optional --send).
        return _once(send=args.send)                          # returns the success/failure code of the one-shot poll function.
    return _daemon()                                          # default mode: start continuous operation as a systemd user daemon (command without flags!).


if __name__ == "__main__":                                    # direct call via `python camwatch.py ...` or through the systemd unit!
    sys.exit(main())
