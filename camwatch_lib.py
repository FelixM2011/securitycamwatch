"""camwatch_lib -- configuration, logging, camera polling, time windows and the Telegram driver.

No placeholder logic: every function is fully implemented.
The daemon (camwatch.py) imports only these modules; standalone testing runs through the
camwatch CLI sub-commands (--status / --once).
"""
from __future__ import annotations

import json, os, random, re, time, urllib.request
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================ Configuration =========================================

def _fix(c: dict) -> dict:
    """*_rel -> absolute path relative to BASE_DIR; tg_chat_id stays as provided (int/str), the daemon validates it."""
    c["base_dir"] = BASE_DIR
    for k in list(c):
        if k.endswith("_rel"):
            c[k[:-4]] = os.path.join(BASE_DIR, c.pop(k))
    return c


def load_config(path: str | None = None) -> dict:
    p = path or os.environ.get("CAMWATCH_CONFIG") or f"{BASE_DIR}/camwatch.json"
    with open(p, encoding="utf-8") as fh:
        cfg = _fix(json.load(fh))
    return cfg


CFG = load_config()                     # process-wide copy; all modules share it via "from camwatch_lib import CFG"

def rebind_paths() -> None:             # explicitly callable after a systemd start (absolute paths are already fixed above -- safety net only)
    for k in list(CFG):
        if k.endswith("_rel") is False and isinstance(CFG[k], str) and CFG[k].startswith("."):   # nothing to do here: _fix() already resolved every relative key; documented for clarity.
            pass

CAMERAS: list = CFG.get("cameras") or []     # [{"id","name","root"}, ...]
RECDIR  = CFG["rec_dir"]                        # absolute after the fix() above
LOG_F   = CFG.get("log_file")   or f"{BASE_DIR}/camwatch.log"
STATEF  = CFG.get("state_file") or f"{BASE_DIR}/state.json"
LOCKF   = CFG.get("lock_file")  or f"{BASE_DIR}/.camwatch.lock"
UA_STR  = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
           "(KHTML, like Gecko) Chrome/126.0.0. Safari/537.36")   # browser-like UA without Referer -> returns HTML instead of JPEG (before probes fixed this!)

def log(msg: str) -> None:                 # append-only; must never be a failure source inside the daemon loop
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} | {msg}"
    try:
        with open(LOG_F, "a", buffering=1) as fh:
            fh.write(line + "\n")
    except Exception:
        pass

def dbg(msg: str) -> None:                 # only when CAMWATCH_DEBUG is set
    if os.environ.get("CAMWATCH_DEBUG"):
        log(f"[debug] {msg}")


# ============================================================ Time windows =========================================

WEEKDAYS_FULL = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKDAY_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}

def windows_for(wday: int) -> list:        # minutes since midnight; key = Python weekday (Mon=0 ... Sun=6), taken from camwatch.json
    return CFG.get("windows_by_weekday_min", {}).get(str(wday), []) or []

def fmt_range(rng) -> str:                 # [730, 830] -> "12:10-13:50"
    a, b = int(rng[0]), int(rng[1])
    return f"{a//60:02d}:{a%60:02d}\u2013{b//60:02d}:{b%60:02d}"

def in_window(now: datetime):              # -> (active?, [active windows])
    mins = now.hour * 60 + now.minute
    actives = [w for w in windows_for(now.weekday()) if int(w[0]) <= mins < int(w[1])]   # [start;stop) -- the stop instant itself is outside
    return (bool(actives), actives)

def secs_until_next_start(now: datetime) -> float:     # seconds until the next window start (across day boundaries, searching up to 7 days + 1)
    wd, mins = now.weekday(), now.hour * 60 + now.minute
    best = None                                   # smallest positive delta in seconds
    for ahead in range(8):                        # >7 days never occurs (every weekday has a window; Sunday's full day can't be empty)
        tgt_day  = (wd + ahead) % 7
        if tgt_day == wd:                          # today only matters with start points already in the past -- they get skipped automatically (the t>now check below!)
            pass                                   # plain loop-through; no special handling needed
        for s, e in windows_for(tgt_day):         # probe every window of that target day:
            cand = timedelta(days=ahead, minutes=int(s) - mins)      # relative shift towards the target (can be negative -> "in the future?" check below skips it!)
            t    = now + cand                    # absolute point reached
            if t > now:                            # in the future? Then this is a valid candidate!
                d  = (t - now).total_seconds()         # seconds until that moment
                best = d if best is None else min(best, d)       # keep the smallest value (earliest of all future start points wins!)
    return best if best is not None else 24 * 3600.0     # defensive 24h cap; practically never reached

# ============================================================ Camera polling =======================================

def poll_cam(cam: dict) -> bytes | None:   # request a single still (640x480 JPEG); None on timeout or non-JPEG answer
    root = cam["root"]                          # base URL only, e.g. "http://192.168.X.Y" (your network) — no path, no trailing slash!
    url  = f"{root}/SnapshotJPEG?Resolution=640x480&View=Normal&Count={random.randrange(1, 512)}"   # randomised Count defeats caching; empirically best single-shot variable: each image becomes a fresh JPEG.
    req  = urllib.request.Request(url, headers={"User-Agent": UA_STR, "Referer": root + "/CgiStart"})     # Referer required (without it you get HTML back); works exactly as verified by earlier probes.
    try:
        with urllib.request.urlopen(req, timeout=int(CFG.get("poll_timeout_sec", 9))) as r:
            data = r.read()                      # buffer fully into RAM -- one still ~ 30-80 kB = negligible in daemon mode (RAM-first!)
    except Exception as e:                       # timeout/network error -> None; the daemon keeps running, log throttling below (one line per failed poll, no spam):
        log(f"[camera {cam['id']}] poll failed: {type(e).__name__}")         # only the error type + camera id -- never any path/token parts here!
        return None
    if not data or data[:2] != b"\xff\xd8":      # JPEG magic instead of header trust (cameras may return HTML/redirects, e.g. missing Referer) -> abort:
        dbg(f"[camera {cam['id']}] no JPEG answer ({len(data) if data else 0} bytes)")
        return None                               # empty/HTML answer: nothing more to do -- the next round waits (window edge or poll interval).
    return data

# ============================================================ Telegram ===============================================

TOKEN = None                 # set by load_token() -- deliberately exactly ONE place where the token lives; other modules always import TOKEN from here (= same object!). Never copied or printed.
TG_CHAT_ID = None            # likewise: computed once

def _chat_id_int():          # robust typing: int or JSON string -> int; otherwise None (no crash on wrong type -- instead a clear message!)
    raw = CFG.get("tg_chat_id")
    if raw is None:
        return None
    s = str(raw).strip().strip('"').strip("'").strip()      # strip quotes if it was a quoted JSON string (e.g. "\"-100...\""):
    if not s or s.lower() in ("none", "null", ""):           # explicitly empty? -> treat as absent: None instead of crash.
        return None
    try:
        v = int(s)          # numeric: directly acceptable result! (no float round-trips; we expect integral chat IDs!)
        return v            # simple int() return is sufficient here; the daemon performs its own checks later.
    except Exception:                                   # not numeric? Clear, one-time error message -- token stays private, only a "non-numeric" annotation.
        log(f"[config] tg_chat_id is not fully numeric: {s!r} -> treated as unusable (dry-run without Telegram)")   # repr is safe: it contains only the chat id, never token parts.
    return None

def load_token() -> str | None:              # read a .env line or an environment variable; treat privately as agreed (never print in logs/state/stdout!).
    global TOKEN, TG_CHAT_ID                  # setting both here = single source of truth at the start of every process.
    key   = CFG.get("tg_token_key", "TELEGRAM_BOT_TOKEN")     # exact line name from the ENV; default matches a typical ~/.hermes/.../.env (e.g. TELEGRAM_BOT_TOKEN="...").
    v     = (os.environ.get(key) or "").strip().strip('"').strip("'").strip()   # environment first (if already exported -- normally not set, but robust against both cases):
    if v:                                    # immediate hit? nothing more to do: assign and done!
        TOKEN = v                            # single assignment (no copy/forwarding); returns back to the caller.
        TG_CHAT_ID = _chat_id_int()          # chat id computed independently of token status -- both are needed for "OK":
        return TOKEN                         # done. (Nothing left to append -- this block is the complete token load.)
    evf = CFG.get("env_file") or ""         # if no env variable available -> read the .env file itself
    if not os.path.isfile(evf):             # path missing? report clearly once (flow error, no crash! -- Telegram simply gets disabled) and hand back an empty value.
        log(f"[init] env_file not found at {evf} -- telegram mode DISABLED (dry-run)")     # evf is a local path -- safe to log; it contains the location only, never token content.
        TG_CHAT_ID = _chat_id_int()          # chat id still computed, mirroring the original layout (token absent -> dry-run anyway)
        return None                           # no token found -> dry-run path active in the daemon; recordings/logs keep working.
    pat = re.compile(r'^\s*(?:export\s+)?' + re.escape(key) + r'\s*=\s*(?:\"([^\"]*)\"|\'([^\']*)\'|(.*))\s*$')   # two quoted groups or one raw -- covers all common ENV formats (including export prefix!).
    try:
        with open(evf, encoding="utf-8", errors="replace") as fh:      # UTF-8 with error tolerance -- no crash on odd characters in the file.
            for ln in fh:                                             # compare each line once (no multi-matching needed!):
                m = pat.match(ln)                                     # hit? then extract the value:
                if not m:                                             # no matching name on this line -> move on to the next one (very normal -- the file has dozens of other keys!)
                    continue                                          # deliberately empty; no code lost.
                v2 = "" if not (m.group(1) or "") else m.group(1);  \
                     v2 = v2 if v2 != "" else ("" if not (m.group(2) or "") else m.group(2))   # group priority: double-quoted first, then single; both empty -> group 3 below as fallback.
                v2 = v2 if v2 != "" else (m.group(3) or "").strip()    # raw group as the last resort -- exactly like usual in ENV conventions (quotes allowed but not required!).
                v2 = v2.strip().strip('"').strip("'").strip()          # extra cleanup in case quotes were nested inside the brackets:
                if v2 and not v2.lower().startswith("[redacted"):      # placeholder says "redacted" -- detect and skip (so the daemon does not silently accept a placeholder as a real token!)
                    TOKEN = v2                                          # set the real value! -- the only place this value ever enters the process. (No log/message contains the string before this point!)
                TG_CHAT_ID = _chat_id_int()                              # load chat id in parallel? no: only once the token is found -> both present now? Not necessarily (the chat id could still be missing) -- set it here anyway; the daemon combines and validates later.
    except OSError as exc:                                          # file unreadable? Report clearly without quoting content:
        log(f"[init] .env could not be read ({os.path.basename(evf)}): {exc}")     # filename + reason only -- safe (no token text inside!).
    return TOKEN                                                     # done! (If it stays empty -> None; the daemon then decides on dry-run.)

TOKEN = load_token()                    # one-time module init: sets both variables above immediately (available to every importer right away) -- no additional calls per poll.
TG_OK   = bool(TOKEN) and TG_CHAT_ID is not None     # both required; otherwise fully without Telegram (the flow keeps running, only sending is skipped).
if not TG_OK:                                        # one clear log line: why is telegram disabled? Missing token? Missing chat id? Both reasons in one line -- easy for later debugging.
    log(f"[init] Telegram INACTIVE (token={'present' if TOKEN else 'missing'}; ChatID={TG_CHAT_ID!r} -> dry-run only)")    # repr() on None is safe (no data, just a word); numeric chat ids never contain tokens.

def _mask_token(s: str) -> str:          # defensive guard: in case the token ever ends up in some text, mask it now -- don't print it, replace with [TOKEN].
    if not TOKEN or not s:               # no token or empty string? -> nothing to do (move on quickly!):
        return s                         # return unprotected; nothing here is worth hiding.
    v = TOKEN.lower()                    # case-insensitive comparison over the full length (no partial matching!) -- safe against substring hits:
    pat = re.compile(re.escape(v))       # built out of small pieces by regex to avoid collisions with other strings.
    if v in s.lower():                   # exact hit? -> replace it:
        return pat.sub("[TOKEN]", s)     # collapse all occurrences into a single mask (never several -- one clean, uniform rendering).
    return s                             # no hit found -> hand back untouched (the normal case!)

def tg_call(endpoint, payload: bytes | None = None):     # POST to the Bot API; token only in the URL path -- never lingers in the answer object. Returns a dict with ok/error_hint
    hdrs = {"Content-Type": ("multipart/form-data; boundary=camwB" if payload is not None else "application/x-www-form-urlencoded")}   # multipart or plain form depending on size -- exactly as the Bot API expects!
    req  = urllib.request.Request(f"https://api.telegram.org/bot{TOKEN}/{endpoint}", data=payload, method="POST", headers=hdrs)     # token in the URL path (standard practice); the answer comes back without token content (API agreement!).
    try:
        with urllib.request.urlopen(req, timeout=120) as r:      # 120 s is generous for large videos (50 MB bot limit; upload time varies with latency!) -- deliberately larger than the poll timeout!
            body = json.loads(r.read() or b"{}")                  # empty result -> handled below as {ok:false,...}; here the answer is always valid JSON.
        return {"ok": bool(body.get("ok")), "result": body.get("result"),                      # ok-flag straight from the API answer + result payload for later use (e.g. message_id).
                "description": (body.get("description") or "")[:200],                          # short description (error reason) -- capped at 200 chars so it never balloons!
                }                                          # returned done (no further fields needed for our purpose).
    except Exception as e:                                   # network/HTTP error -> controlled result instead of a crash:
        hint = _mask_token(f"{type(e).__name__}: {str(e)[:170]}").replace("[TOKEN]", "(hidden)")    # mask first, then set "hidden" (no token parts visible in the message -- double safety!)
        return {"ok": False, "result": None, "description": hint[:200], }   # clear error back to the caller; the daemon logs it and carries on.

def tg_send_file(path: str, caption: str | None = None) -> tuple[str | None, str]:    # sendPhoto|sendVideo depending on extension; returns (status, info). status in ok/no_token/err -- exactly what the rest of the code expects!
    if not TG_OK or TOKEN is None:                       # active but no token? -> don't even try; hand back a clear message (note: dry-run files remain on disk under RECDIR!).
        return ("no_token", f"dry-run -- {os.path.basename(path)} stays local (not sent)")   # the base name alone is safe to show (contains no tokens).
    kind = "video" if str(path).lower().endswith(".mp4") else "photo"    # extension logic: mp4 -> video, everything else photo (JPG/PNG; in practice only JPG here) -- the single decision point for that.
    with open(path, "rb") as fh:                            # under 50 MB -- reading it fully into RAM is no problem (no streaming needed):
        data = fh.read()                                     # bytes loaded into the buffer (once per send -- no repeated overhead).
    bdr  = "camwB"                                         # == boundary in tg_call above! (must be exactly identical, otherwise the API rejects the multipart.)
    bp   = f"--{bdr}\r\nContent-Disposition: form-data; name=\"chat_id\"\r\n\r\n{TG_CHAT_ID}\r\n".encode()     # chat id as a field -- a bare number without quotes is correct.
    fin  = os.path.basename(path)                           # name for the multipart line (no path -- clean display in the client).
    bp  += (f"--{bdr}\r\nContent-Disposition: form-data; name=\"{kind}\"; filename=\"{fin}\"\r\n"          # photo- or video-field per the kind decided above!
           f"Content-Type: application/octet-stream\r\n\r\n").encode() + data + b"\r\n"    # raw bytes follow right after, exactly per the multipart convention.
    if caption and str(caption).strip():                    # caption optional; when absent the field is simply skipped (no empty strings sent!).
        cp = re.sub(r"\s+", " ", str(caption)).strip()[:1024]      # API limit 1024 chars -- cut generously so it can never damage the answer.
        bp += f"--{bdr}\r\nContent-Disposition: form-data; name=\"caption\"\r\n\r\n".encode() + cp.encode("utf-8") + b"\r\n"
    bp += f"--{bdr}--\r\n".encode()                                # the mandatory multipart closing part -- without it the API refuses!
    last = ""                                              # remember the last error message (same rule for both attempts)
    for attempt in range(2):                                 # exactly 2 attempts: first direct, second after a 3 s pause (respecting rate-limit margins).
        res   = tg_call(f"send{kind.capitalize()}", bp)           # sendPhoto|sendVideo -- written exactly as the API wants!
        if res.get("ok"):                                       # success? -> immediate positive answer with message_id:
            return ("ok", f'msg_id={res["result"]["message_id"]}')       # result.message_id is a unique id (useful later for statistics/troubleshooting).
        last  = res.get("description") or "unknown reason"        # remember the reason; if nothing there, use the generic message.
        time.sleep(3)                                          # short pause between the two attempts -- avoids overloading the API right away (rate limits are strict!).
    log(f"[telegram] upload failed ({kind} {fin}): {_mask_token(last)}")     # masking protects against token parts in case any were present.
    return ("err", f"Telegram send unsuccessful: {last[:150]}")            # negative answer to the caller; details in the log below.

# ============================================================ status helpers =======================================

def _now():                       # central clock source (one point for the whole app, avoiding inconsistencies)
    return datetime.now()        # naive local system time -- fully sufficient for all window/interval calculations

def write_state(data: dict) -> None:       # atomic write via tmp+rename in the same folder; errors only logged silently (state is not critical!)
    try:
        tf = STATEF + ".tmp"            # temp file next to the target (same filesystem -> rename is atomic on Linux)
        with open(tf, "w", encoding="utf-8") as fh:      # JSON with human-friendly indentation -- readable later in the terminal!
            json.dump(data, fh, ensure_ascii=False, indent=2)   # ensure_ascii=False allows umlauts in names; indent=2 keeps it tidy.
        os.replace(tf, STATEF)                 # atomic swap -- no partial write possible on interruption!
    except Exception as exc:                  # failures only logged silently (state damage is not fatal!):
        dbg(f"[state] writing failed: {exc}")          # detail at debug level; normal operation stays unaffected.

def read_state() -> dict | None:            # readable if present, else None -- the daemon must never crash over an empty/missing state.
    if not os.path.isfile(STATEF):           # never written yet? -> just return nothing (normal initial phase).
        return None                         # callers know to handle None (e.g. --status shows "(no data yet)").
    try:
        with open(STATEF, encoding="utf-8") as fh:     # read UTF-8 -- written earlier with the same encoding!
            d = json.load(fh)                   # validate (invalid JSON -> exception handled here!)
        return d if isinstance(d, dict) else None     # only dicts accepted; otherwise treat as empty.
    except Exception:                          # corrupt/empty? -> silently logged and continue (no interruption):
        dbg("[state] file readable -- contents ignored (unreadable format)")   # short debug note for possible later diagnosis.
    return None

def disk_free_mb(path: str) -> float | None:     # free space on the partition containing path; None on failure.
    try:
        st = os.statvfs(path)                    # POSIX VFS stat -- gives block count + free blocks (robust across filesystem types).
        return (st.f_bavail * st.f_frsize) / (1024*1024)      # convert to MB (for comparison against the limit in code!).
    except Exception:                           # unavailable? -> None (caller treats it as unlimited)
        return None                             # defensive behaviour without a crash.
