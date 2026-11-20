#!/usr/bin/env python3
"""Viral cross-reference: find clips other people already made from the same
stream, video or creator, and locate them on the source timeline.

  viral.py audio.wav --vod URL [--creator NAME] [--query "extra search"]...
           [--yt-channel URL]... [--max-refs 25] [--work DIR] > viral.json

A moment that someone else already clipped (and that got views) is the
strongest outside signal that it works for a newcomer. This script:

1. Reads the VOD's metadata (title, creator, date, id) with yt-dlp.
2. Collects reference clips:
   * Twitch (official Helix API): clips cut from THIS VOD, with their exact
     VOD offset, view count and title. Needs a Twitch developer app:
     TWITCH_CLIENT_ID in the environment, and its app access token either in
     TWITCH_TOKEN or injected for api.twitch.tv by the session's credential
     proxy. Skipped with a note when not configured.
   * YouTube (yt-dlp search, no key): Shorts / clip-channel uploads for the
     creator and the VOD's topic, plus any --yt-channel pages (e.g. a clip
     channel's /shorts tab). Only uploads up to --max-ref-dur seconds that were
     posted on or after the stream date are kept.
3. Locates every reference without a known offset by matching its audio
   against the stream audio (onset envelopes in 8 bands at 10 fps, normalised
   cross-correlation over the whole stream). Long references (recaps,
   compilations) are probed in several places, so each segment they reuse is
   found separately. A match needs a clear peak (z >= --min-z) to count.
4. Merges overlapping matches into "moments", scored by views and the number
   of independent re-uploads, and pools the hashtags/tags those clips used,
   weighted by views (post_kit.py uses the pool for captions).

Output (stdout, JSON):
  {"vod": {...}, "refs": [...every reference with its match...],
   "moments": [{"t0", "t1", "hms", "score", "views", "sources": [...]}],
   "hints": [...most-viewed references that could not be located: titles to look for],
   "hashtags": [["gta6", weight, uploads_using_it], ...], "notes": [...]}

Feed it to scout.py --viral viral.json (boosts and force-includes those
moments) and to post_kit.py --viral viral.json (hashtag pool).
The stream audio's features are cached as audio.wav.feat.npy.
"""
import argparse, json, math, os, re, subprocess, sys, tempfile, time, urllib.parse, urllib.request, wave
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch import ff  # noqa: E402  (ffmpeg argv with -nostdin and the proxy)

FPS = 10                      # feature frames per second
HOP = 16000 // FPS
NB = 8                        # bands


# ---------------------------------------------------------------- features
def band_edges(sr=16000, nfft=2048):
    f = np.fft.rfftfreq(nfft, 1 / sr)
    e = np.geomspace(150, 5000, NB + 1)
    return [(f >= e[i]) & (f < e[i + 1]) for i in range(NB)]


def features_from_samples(x):
    """Onset envelope per band: first difference of log band energy (gain- and
    EQ-invariant), half-wave rectified, z-normalised per band."""
    lb = raw_logbands(np.concatenate([x, np.zeros(2048, np.float32)]))
    if len(lb) < 4:
        return np.zeros((0, NB), np.float32)
    return onset_norm(lb)


def onset_norm(lb):
    d = np.maximum(0, np.diff(lb, axis=0, prepend=lb[:1]))
    d -= d.mean(0); d /= d.std(0) + 1e-6
    return d.astype(np.float32)


def read_wav16(path):
    w = wave.open(path)
    assert w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == 16000, \
        f"{path}: need 16 kHz mono 16-bit wav"
    return w


def stream_features(path):
    cache = path + ".feat.npy"
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(path):
        return np.load(cache)
    w = read_wav16(path)
    N = w.getnframes(); chunk = 16000 * 600
    parts = []
    for c0 in range(0, N, chunk):
        x = np.frombuffer(w.readframes(chunk), np.int16).astype(np.float32) / 32768
        n = len(x) // HOP
        parts.append(raw_logbands(np.concatenate([x, np.zeros(2048, np.float32)]))[:n])
        print(f"\rviral: stream features {min(N, c0 + chunk) / 16000 / 3600:5.2f} h", end="", file=sys.stderr)
    print(file=sys.stderr)
    d = onset_norm(np.concatenate(parts))
    np.save(cache, d)
    return d


def raw_logbands(x):
    nfft = 2048
    n = max(0, (len(x) - nfft) // HOP + 1)
    bands = band_edges(nfft=nfft)
    win = np.hanning(nfft).astype(np.float32)
    out = np.zeros((n, NB), np.float32)
    for b0 in range(0, n, 3000):
        idx = np.arange(b0, min(n, b0 + 3000))
        fr = np.stack([x[s:s + nfft] for s in idx * HOP]) * win
        pw = np.abs(np.fft.rfft(fr, nfft)) ** 2
        out[idx] = np.log(np.stack([pw[:, m].sum(1) for m in bands], 1) + 1e-8)
    return out


class Matcher:
    """Normalised cross-correlation of a probe (P x NB) against the stream
    (F x NB). Stream spectra are computed once; one inverse FFT per probe."""

    def __init__(self, S):
        self.S = S; self.F = len(S)
        self.n = 1 << int(math.ceil(math.log2(self.F + 64 * FPS)))
        self.SF = [np.fft.rfft(S[:, b], self.n).astype(np.complex64) for b in range(NB)]
        e = (S ** 2).sum(1)
        self.cum = np.concatenate([[0], np.cumsum(e, dtype=np.float64)])

    def locate(self, P):
        L = len(P)
        if L < 5 * FPS or L >= self.F:
            return None
        acc = np.zeros(self.n // 2 + 1, np.complex64)
        for b in range(NB):
            acc += self.SF[b] * np.conj(np.fft.rfft(P[:, b], self.n).astype(np.complex64))
        corr = np.fft.irfft(acc, self.n)[: self.F - L + 1]
        win_e = self.cum[L:] - self.cum[:-L]
        pe = float((P ** 2).sum())
        ncc = corr / np.sqrt(np.maximum(win_e[: len(corr)], 1e-6) * pe + 1e-9)
        k = int(np.argmax(ncc)); peak = float(ncc[k])
        lo, hi = max(0, k - 5 * FPS), k + 5 * FPS          # exclude the peak's own neighbourhood
        rest = np.concatenate([ncc[:lo], ncc[hi:]])
        med = float(np.median(rest)); mad = float(np.median(np.abs(rest - med))) * 1.4826 + 1e-6
        return {"t": k / FPS, "ncc": round(peak, 3), "z": round((peak - med) / mad, 1)}


# ---------------------------------------------------------------- sources
def run(cmd, timeout=180):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "timeout"


def ytdlp_json(url, *extra, cookies=None, timeout=180):
    cmd = ["yt-dlp", "--no-warnings", "-J", "--skip-download"] + (["--cookies", cookies] if cookies else []) + list(extra) + [url]
    rc, out, err = run(cmd, timeout)
    if rc:
        return None, err.strip()[-300:]
    try:
        return json.loads(out), None
    except ValueError:
        return None, "bad json"


def helix(path, params, notes):
    cid = os.environ.get("TWITCH_CLIENT_ID")
    if not cid:
        return None
    tok = os.environ.get("TWITCH_TOKEN", "proxy-injected")
    url = "https://api.twitch.tv/helix/" + path + "?" + urllib.parse.urlencode(params, doseq=True)
    req = urllib.request.Request(url, headers={"Client-Id": cid, "Authorization": f"Bearer {tok}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except Exception as e:  # noqa: BLE001
        notes.append(f"twitch api {path}: {type(e).__name__}: {str(e)[:160]}")
        return None


def twitch_clips(vod, notes):
    if not os.environ.get("TWITCH_CLIENT_ID"):
        notes.append("twitch clips skipped: set TWITCH_CLIENT_ID (+ app token in TWITCH_TOKEN or the "
                     "session's credential for api.twitch.tv) to include clips cut from this VOD")
        return []
    login = vod.get("uploader_id") or vod.get("uploader") or ""
    u = helix("users", {"login": login.lower()}, notes)
    if not u or not u.get("data"):
        notes.append(f"twitch: user {login!r} not found"); return []
    bid = u["data"][0]["id"]
    ts = vod.get("timestamp") or time.time() - 86400
    start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts - 3600))
    end = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts + (vod.get("duration") or 0) + 3600))
    out, cursor = [], None
    for _ in range(20):                                  # up to 2000 clips
        p = {"broadcaster_id": bid, "started_at": start, "ended_at": end, "first": 100}
        if cursor: p["after"] = cursor
        r = helix("clips", p, notes)
        if not r: break
        for c in r.get("data", []):
            if str(c.get("video_id")) != str(vod["id"]).lstrip("v"):
                continue
            ref = {"source": "twitch_clip", "url": c["url"], "title": c["title"], "views": c.get("view_count", 0),
                   "duration": c.get("duration"), "uploader": c.get("creator_name"), "tags": [],
                   "upload_date": c.get("created_at", "")[:10].replace("-", "")}
            if c.get("vod_offset") is not None:
                ref["match"] = {"t": float(c["vod_offset"]), "z": None, "exact": True,
                                "spans": [[float(c["vod_offset"]), float(c["vod_offset"]) + float(c.get("duration") or 30)]]}
            out.append(ref)
        cursor = (r.get("pagination") or {}).get("cursor")
        if not cursor: break
    notes.append(f"twitch: {len(out)} clips from this VOD")
    return out


STOP = set("the a an and or of to in on for with at by is it this that day part live stream vod my your we i "
           "new full first last 24 7 24/7 marathon giveaway giveaways ps5".split())


def topic_words(title, creator):
    w = [x for x in re.findall(r"[A-Za-z0-9]+", title or "") if x.lower() not in STOP and x.lower() != (creator or "").lower()]
    return " ".join(w[:4])


def youtube_refs(vod, creator, queries, channels, per_query, max_dur, cookies, notes):
    topic = topic_words(vod.get("title", ""), creator)
    qs = list(queries) or [f"{creator} {topic}".strip(), f"{creator} shorts", f"{creator} funny moments",
                           f"{creator} clips {topic}".strip()]
    ids = {}
    for q in qs:
        d, err = ytdlp_json(f"ytsearch{per_query}:{q}", "--flat-playlist", cookies=cookies, timeout=120)
        if not d:
            notes.append(f"youtube search {q!r} failed: {err}"); continue
        for e in d.get("entries") or []:
            if e.get("id") and (e.get("duration") or 0) <= max_dur and e.get("id") != vod.get("id"):
                ids.setdefault(e["id"], (q, e))
    for ch in channels:
        d, err = ytdlp_json(ch, "--flat-playlist", "--playlist-end", str(per_query * 2), cookies=cookies, timeout=120)
        if not d:
            notes.append(f"channel {ch} failed: {err}"); continue
        for e in d.get("entries") or []:
            if e.get("id") and (e.get("duration") or 0) <= max_dur:
                ids.setdefault(e["id"], (ch, e))
    notes.append(f"youtube: {len(ids)} candidate uploads from {len(qs)} searches + {len(channels)} channels")

    def meta(vid):
        d, err = ytdlp_json(f"https://www.youtube.com/watch?v={vid}", cookies=cookies, timeout=90)
        return vid, d, err
    refs = []
    vod_date = vod.get("upload_date") or ""
    blocked = 0
    with ThreadPoolExecutor(4) as ex:
        for vid, d, err in ex.map(meta, list(ids)):
            if not d:
                # YouTube often demands sign-in from datacenter IPs. Keep the search
                # result (title, views, channel) as an unlocated hint; it can't be
                # audio-matched or date-checked without --cookies.
                blocked += "sign in" in (err or "").lower()
                q, e = ids[vid]
                refs.append({"source": "youtube", "url": f"https://www.youtube.com/watch?v={vid}",
                             "title": e.get("title") or "", "views": e.get("view_count") or 0,
                             "duration": e.get("duration"), "uploader": e.get("channel") or e.get("uploader"),
                             "upload_date": None, "tags": [], "hashtags": re.findall(r"#(\w+)", e.get("title") or ""),
                             "query": q, "search_only": True})
                continue
            if vod_date and (d.get("upload_date") or "99999999") < vod_date:
                continue                                   # posted before the stream: can't be from it
            if (d.get("duration") or 0) > max_dur:
                continue
            refs.append({"source": "youtube", "url": d.get("webpage_url"), "title": d.get("title", ""),
                         "views": d.get("view_count") or 0, "duration": d.get("duration"),
                         "uploader": d.get("channel") or d.get("uploader"), "upload_date": d.get("upload_date"),
                         "tags": (d.get("tags") or [])[:30],
                         "hashtags": re.findall(r"#(\w+)", (d.get("title") or "") + " " + (d.get("description") or ""))[:30],
                         "query": ids[vid][0]})
    if blocked:
        notes.append(f"youtube asked for sign-in on {blocked} uploads: kept their search titles/views as "
                     "unlocated hints; pass --cookies cookies.txt (exported from a logged-in browser) to "
                     "audio-match them and read their tags")
    return refs


def ref_audio(url, out, max_dur, cookies):
    rc, u, err = run(["yt-dlp", "--no-warnings", "-g", "-f", "bestaudio/best"] + (["--cookies", cookies] if cookies else []) + [url], 90)
    if rc or not u.strip():
        return False
    rc, _, _ = run(ff("-i", u.strip().splitlines()[-1], "-t", str(max_dur), "-vn", "-ac", "1", "-ar", "16000", out), 300)
    return rc == 0 and os.path.exists(out)


def probes_for(L_sec):
    """(start, length) probes in seconds: short refs whole-ish, long refs sampled."""
    plen = min(15.0, max(6.0, L_sec - 1))
    if L_sec <= 75:
        starts = np.arange(0, max(L_sec - plen, 0) + 0.01, 10.0)
    else:
        starts = np.linspace(2, L_sec - plen - 2, 20)
    return [(float(s), plen) for s in starts]


def match_ref(M, ref_wav, min_z, min_ncc=0.35):
    x = np.frombuffer(read_wav16(ref_wav).readframes(10 ** 9), np.int16).astype(np.float32) / 32768
    R = features_from_samples(x)
    L = len(R) / FPS
    hits = []
    for s, pl in probes_for(L):
        P = R[int(s * FPS): int((s + pl) * FPS)]
        m = M.locate(P)
        if m and m["z"] >= min_z and m["ncc"] >= min_ncc:
            hits.append({"ref_t": s, "t": m["t"], "len": pl, "z": m["z"], "ncc": m["ncc"]})
    if not hits:
        return None
    # chain hits whose stream position advances with the reference position (same segment)
    hits.sort(key=lambda h: h["t"])
    spans = []
    for h in hits:
        off = h["t"] - h["ref_t"]
        if spans and abs(off - spans[-1]["off"]) < 3 and h["t"] - spans[-1]["t1"] < 30:
            spans[-1]["t1"] = h["t"] + h["len"]; spans[-1]["z"] = max(spans[-1]["z"], h["z"])
        else:
            spans.append({"off": off, "t0": h["t"], "t1": h["t"] + h["len"], "z": h["z"]})
    best = max(spans, key=lambda s: s["z"])
    return {"t": round(best["t0"], 1), "z": best["z"], "exact": False,
            "spans": [[round(s["t0"], 1), round(s["t1"], 1)] for s in spans]}


# ---------------------------------------------------------------- main
def hms(t):
    t = int(t); return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", help="the stream's 16 kHz mono audio (fetch.py audio)")
    ap.add_argument("--vod", required=True, help="VOD / video URL the audio came from")
    ap.add_argument("--offset", type=float, default=0, help="stream time of the wav's first sample")
    ap.add_argument("--creator", help="creator name for searches (default: VOD uploader)")
    ap.add_argument("--query", action="append", default=[], help="extra YouTube search (repeatable); replaces the defaults")
    ap.add_argument("--yt-channel", action="append", default=[], help="channel/playlist URL to scan, e.g. a clip channel's /shorts")
    ap.add_argument("--per-query", type=int, default=15)
    ap.add_argument("--max-refs", type=int, default=25, help="most-viewed references to audio-match")
    ap.add_argument("--max-ref-dur", type=float, default=1200, help="ignore uploads longer than this (s)")
    ap.add_argument("--min-z", type=float, default=7.0, help="match peak must stand this many MADs above the rest")
    ap.add_argument("--min-ncc", type=float, default=0.35, help="and correlate at least this well (true matches ~0.45-0.97, noise ~0.15)")
    ap.add_argument("--work", default=None, help="scratch dir for reference audio")
    ap.add_argument("--cookies")
    ap.add_argument("--no-youtube", action="store_true")
    a = ap.parse_args()

    notes = []
    vod, err = ytdlp_json(a.vod, cookies=a.cookies)
    if not vod:
        sys.exit(f"cannot read {a.vod}: {err}")
    creator = a.creator or vod.get("uploader") or vod.get("channel") or ""
    vinfo = {"id": vod.get("id"), "title": vod.get("title"), "creator": creator, "url": a.vod,
             "upload_date": vod.get("upload_date"), "duration": vod.get("duration")}

    refs = []
    if "twitch" in (vod.get("extractor") or "").lower():
        refs += twitch_clips(vod, notes)
    if not a.no_youtube:
        refs += youtube_refs(vod, creator, a.query, a.yt_channel, a.per_query, a.max_ref_dur, a.cookies, notes)

    # audio-match the most-viewed references that don't carry an offset
    todo = sorted([r for r in refs if "match" not in r and not r.get("search_only")],
                  key=lambda r: -(r.get("views") or 0))[: a.max_refs]
    if todo:
        S = stream_features(a.wav)
        M = Matcher(S)
        work = a.work or tempfile.mkdtemp(prefix="clipify_viral_")
        os.makedirs(work, exist_ok=True)

        def dl(r):
            p = os.path.join(work, re.sub(r"\W+", "_", r["url"])[-60:] + ".wav")
            ok = os.path.exists(p) or ref_audio(r["url"], p, a.max_ref_dur, a.cookies)
            return r, p if ok else None
        with ThreadPoolExecutor(3) as ex:
            for i, (r, p) in enumerate(ex.map(dl, todo)):
                if not p:
                    r["match"] = None; r["match_note"] = "audio download failed"; continue
                m = match_ref(M, p, a.min_z, a.min_ncc)
                if m:
                    m["t"] += a.offset; m["spans"] = [[s0 + a.offset, s1 + a.offset] for s0, s1 in m["spans"]]
                r["match"] = m
                print(f"\rviral: matched {i + 1}/{len(todo)}", end="", file=sys.stderr)
        print(file=sys.stderr)

    # merge matched spans into moments
    spans = []
    for r in refs:
        m = r.get("match")
        if not m: continue
        for s0, s1 in m["spans"]:
            spans.append((s0, max(s1, s0 + 5), r))
    spans.sort(key=lambda s: s[0])
    moments = []
    for s0, s1, r in spans:
        # same moment = starts within 30 s of the moment's first clip (no chaining:
        # a busy stream has overlapping clips back to back for minutes)
        if moments and s0 - moments[-1]["t0"] <= 30:
            mo = moments[-1]; mo["t1"] = max(mo["t1"], s1)
        else:
            mo = {"t0": s0, "t1": s1, "refs": []}; moments.append(mo)
        mo["refs"].append(r)
    out_m = []
    for mo in moments:
        uniq = list({r["url"]: r for r in mo["refs"]}.values())
        views = sum(r.get("views") or 0 for r in uniq)
        top = max((r.get("views") or 0) for r in uniq)
        # views lead; more independent clips/uploads of the same moment add a bonus
        score = math.log10(views + 10) + 0.5 * math.log2(len(uniq))
        out_m.append({"t0": round(mo["t0"], 1), "t1": round(mo["t1"], 1), "hms": hms(mo["t0"]),
                      "score": round(score, 2), "views": views, "top_views": top, "uploads": len(uniq),
                      "sources": [{"title": r["title"], "views": r.get("views"), "url": r["url"], "source": r["source"]}
                                  for r in sorted(uniq, key=lambda r: -(r.get("views") or 0))]})
    out_m.sort(key=lambda m: -m["score"])

    # hashtag pool weighted by views (tags + #hashtags of every reference, matched or not)
    pool, uses = Counter(), Counter()
    for r in refs:
        w = math.log10((r.get("views") or 0) + 10)
        tags = {re.sub(r"[^0-9a-z_]", "", x.lower().replace(" ", "")) for x in (r.get("hashtags") or []) + (r.get("tags") or [])}
        for t in tags:
            if 2 < len(t) <= 30:
                pool[t] += w; uses[t] += 1
    notes.append(f"{sum(1 for r in refs if r.get('match'))} of {len(refs)} references located; {len(out_m)} moments")
    for n in notes:
        print(f"viral: {n}", file=sys.stderr)
    # unlocated references (search-only, or no audio match): title hints to look for in the digest
    hints = [{"title": r["title"], "views": r.get("views"), "duration": r.get("duration"),
              "uploader": r.get("uploader"), "url": r["url"]}
             for r in sorted(refs, key=lambda r: -(r.get("views") or 0)) if not r.get("match")][:40]
    json.dump({"vod": vinfo, "refs": refs, "moments": out_m, "hints": hints,
               "hashtags": [[t, round(w, 2), uses[t]] for t, w in pool.most_common(60)], "notes": notes},
              sys.stdout, indent=1)


if __name__ == "__main__":
    main()
