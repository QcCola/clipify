#!/usr/bin/env python3
"""Scout a long stream recording for clip candidates (stream mode).

  scout.py audio.wav [--chat chat.json] [--per-hour 4] [--min-gap 90]
           [--windows OUTDIR --pre 35 --post 25] > candidates.json

audio.wav: 16 kHz mono int16 (fetch.py audio writes this). Hours of audio are
processed in 10-minute chunks, so memory stays flat.

Signals, scored per second and each normalised against the whole stream
(robust z-score: median / MAD) so one loud streamer doesn't drown everything:

  reaction   loudness above the surrounding 5 minutes (shouts, hype)
  payoff     quiet setup followed by a burst: the comedic-timing shape
             (mean of the 8 s before vs. peak of the 3 s after)
  laugh      laugh-like rhythm: 3.5-7 Hz syllable pulsing in the 1-4 kHz
             voice band ("ha-ha-ha"), weighted by loudness (heuristic)
  game       density of low-band impacts (crashes, gunfire, explosions,
             brakes) above the local norm: things happening in the game
  chat       optional: message-rate spike and laugh/hype-emote density,
             shifted 6 s earlier because chat reacts late

Muted stretches (Twitch DMCA mutes) and the 30 s around them are ignored.
Candidates are picked with a minimum gap and a per-hour floor so the whole
recording is covered, not just its loudest hour. Each candidate lists its
dominant signal ("kind") so the batch can lead with different kinds of moment.

--windows writes OUTDIR/w_<sec>.wav around each candidate (no ffmpeg loop,
so nothing can eat stdin) ready for one batched whisper call.

Chat formats accepted: TwitchDownloader JSON ({"comments":[{"content_offset_seconds",
"message":{"body"}}]}), or a list of {"t": seconds, "text": "..."}, or CSV/TSV
lines "seconds,text".
"""
import argparse, json, os, re, sys, wave
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("wav")
ap.add_argument("--chat")
ap.add_argument("--per-hour", type=int, default=4, help="min candidates per hour of stream (coverage floor)")
ap.add_argument("--top", type=int, default=0, help="total candidates (default: 6 per hour)")
ap.add_argument("--min-gap", type=float, default=90, help="seconds between candidates")
ap.add_argument("--windows"); ap.add_argument("--pre", type=float, default=35); ap.add_argument("--post", type=float, default=25)
ap.add_argument("--dump", help="save per-second signals to this .npz (debugging)")
ap.add_argument("--offset", type=float, default=0, help="absolute stream time of the wav's first sample")
a = ap.parse_args()

w = wave.open(a.wav)
assert w.getnchannels() == 1 and w.getsampwidth() == 2, "need mono 16-bit wav"
SR = w.getframerate(); N = w.getnframes(); S = N // SR
HOP = SR // 50                      # 20 ms frames -> 50 fps
NFFT = 1 << int(np.ceil(np.log2(HOP)))
freqs = np.fft.rfftfreq(NFFT, 1 / SR)
LOW = (freqs >= 50) & (freqs < 250)
VOICE = (freqs >= 1000) & (freqs < 4000)
win = np.hanning(HOP).astype(np.float32)

db = np.zeros(S, np.float32)        # loudness per second
laugh = np.zeros(S, np.float32)
onsets = np.zeros(S, np.float32)

CH = 600
prev_low = None
for c0 in range(0, S, CH):
    k = min(CH, S - c0)
    x = np.frombuffer(w.readframes(k * SR), np.int16).astype(np.float32)[: k * SR] / 32768
    fr = x.reshape(-1, HOP) * win
    mag = np.abs(np.fft.rfft(fr, NFFT))
    pw = (mag ** 2)
    tot = 10 * np.log10(pw.sum(1) + 1e-10)
    voice = 10 * np.log10(pw[:, VOICE].sum(1) + 1e-10)
    lowlog = np.log(mag[:, LOW] + 1e-6)
    if prev_low is None: prev_low = lowlog[:1]
    flux = np.maximum(0, np.diff(np.vstack([prev_low[-1:], lowlog]), axis=0)).sum(1)
    prev_low = lowlog[-1:]
    thr = np.median(flux) + 6 * (np.median(np.abs(flux - np.median(flux))) + 1e-6)
    on = (flux > thr).astype(np.float32)
    db[c0:c0 + k] = tot.reshape(k, 50).mean(1)
    onsets[c0:c0 + k] = on.reshape(k, 50).sum(1)
    # laugh rhythm: modulation spectrum of the voice-band envelope over 2 s windows
    env = voice.reshape(k, 50)
    for i in range(k):
        seg = env[max(0, i - 1):i + 1].ravel()
        if len(seg) < 100: seg = np.pad(seg, (100 - len(seg), 0), mode="edge")
        seg = seg - seg.mean()
        sp = np.abs(np.fft.rfft(seg * np.hanning(100))) ** 2
        f = np.fft.rfftfreq(100, 1 / 50)
        band = sp[(f >= 3.5) & (f <= 7)].sum(); allb = sp[(f >= 0.5) & (f <= 12)].sum() + 1e-9
        laugh[c0 + i] = band / allb
    print(f"\rscouting {min(S, c0 + k) / 3600:5.2f} / {S / 3600:.2f} h", end="", file=sys.stderr)
print(file=sys.stderr)

# --- muted / silent stretches
silent = db < -60
near_mute = np.convolve(silent.astype(np.float32), np.ones(61), mode="same") > 0

def local_median(v, half=150, blk=10):
    nb = len(v) // blk + 1
    b = np.array([np.median(v[i * blk:(i + 1) * blk][~silent[i * blk:(i + 1) * blk]])
                  if (~silent[i * blk:(i + 1) * blk]).any() else np.nan for i in range(nb)])
    h = half // blk
    out = np.array([np.nanmedian(b[max(0, j - h):j + h]) if np.isfinite(b[max(0, j - h):j + h]).any() else 0
                    for j in range(nb)])
    return np.repeat(out, blk)[:len(v)]

def rz(v):
    """Robust z-score. Spread = max(MAD, quarter of the 50-95th percentile range) so
    sparse count signals (mostly zero) don't explode."""
    u = v[~near_mute] if (~near_mute).any() else v
    m = np.median(u)
    d = max(1.4826 * np.median(np.abs(u - m)), (np.percentile(u, 95) - m) / 4, 1e-3)
    return np.clip((v - m) / d, -3, 6)

sm = np.convolve(db, np.ones(3) / 3, mode="same")
reaction = sm - local_median(sm)
pre = np.convolve(db, np.ones(8) / 8, mode="full")[:S]           # mean of previous 8 s (ending at t)
pre = np.concatenate([np.full(1, pre[0]), pre[:-1]])
post = np.array([db[t:t + 3].max() if t < S else db[-1] for t in range(S)])
lm = local_median(sm)
# comedic-timing shape: a quiet beat then a burst, AND the burst must be loud for this
# stretch of stream (otherwise silence -> normal talking would score)
payoff = np.minimum(post - pre, 2 * (post - lm))
laugh_w = laugh * np.clip((sm - np.median(sm[~silent])) / 10 + 1, 0, 2)
game_d = np.convolve(onsets, np.ones(3), mode="same")
game = game_d - local_median(game_d)

feats = {"reaction": rz(reaction), "payoff": rz(payoff), "laugh": rz(laugh_w), "game": rz(game)}
weights = {"reaction": 1.0, "payoff": 1.0, "laugh": 0.8, "game": 0.7}

# --- chat
if a.chat:
    raw = open(a.chat).read()
    msgs = []
    try:
        j = json.loads(raw)
        if isinstance(j, dict) and "comments" in j:
            for c in j["comments"]:
                body = c.get("message", {}).get("body") or " ".join(f.get("text", "") for f in c.get("message", {}).get("fragments", []))
                msgs.append((float(c["content_offset_seconds"]), body))
        else:
            msgs = [(float(m["t"]), m.get("text", "")) for m in j]
    except ValueError:
        for line in raw.splitlines():
            p = re.split(r"[,\t]", line, 1)
            try: msgs.append((float(p[0]), p[1] if len(p) > 1 else ""))
            except ValueError: pass
    LAUGH = re.compile(r"(LUL|KEKW|OMEGALUL|ICANT|LMAO+|LOL+|xd+|😂|🤣|💀|HAHA+|Kappa|PogU?|POG+|W+\b|Clueless|monkaS)", re.I)
    rate = np.zeros(S, np.float32); lol = np.zeros(S, np.float32)
    for t, body in msgs:
        i = int(t - a.offset) - 6
        if 0 <= i < S:
            rate[i] += 1; lol[i] += len(LAUGH.findall(body))
    r5 = np.convolve(rate, np.ones(5), mode="same"); l5 = np.convolve(lol, np.ones(5), mode="same")
    feats["chat"] = rz(r5 - local_median(r5)) * 0.5 + rz(l5) * 0.5
    weights["chat"] = 1.5
    print(f"chat: {len(msgs)} messages", file=sys.stderr)

score = sum(weights[k] * np.clip(v, 0, None) for k, v in feats.items())
score = np.convolve(score, np.ones(3) / 3, mode="same")
score[near_mute] = -1

if a.dump: np.savez(a.dump, score=score, db=db, **feats)

# --- pick: per-hour floor first, then best remaining
top = a.top or max(1, int(np.ceil(S / 3600 * 6)))
picked = []
def ok(t): return score[t] > 0 and all(abs(t - p) >= a.min_gap for p in picked)
for h0 in range(0, S, 3600):
    order = h0 + np.argsort(-score[h0:h0 + 3600])
    n = 0
    for t in order:
        if n >= a.per_hour: break
        if ok(int(t)): picked.append(int(t)); n += 1
for t in np.argsort(-score):
    if len(picked) >= top: break
    if ok(int(t)): picked.append(int(t))

def hms(t):
    t = int(t); return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}"

cands = []
for t in sorted(picked):
    parts = {k: round(float(weights[k] * max(0, v[t])), 2) for k, v in feats.items()}
    kind = max(parts, key=parts.get)
    cands.append({"t": t + a.offset, "hms": hms(t + a.offset), "score": round(float(score[t]), 2),
                  "kind": kind, "signals": parts, "db": round(float(db[t]), 1)})
cands.sort(key=lambda c: -c["score"])
for r, c in enumerate(cands, 1): c["rank"] = r

if a.windows:
    os.makedirs(a.windows, exist_ok=True)
    for c in cands:
        t = c["t"] - a.offset
        s0 = int(max(0, t - a.pre)); s1 = int(min(S, t + a.post))
        w.setpos(s0 * SR)
        data = w.readframes((s1 - s0) * SR)
        p = os.path.join(a.windows, f"w_{int(c['t'])}.wav")
        o = wave.open(p, "wb"); o.setnchannels(1); o.setsampwidth(2); o.setframerate(SR); o.writeframes(data); o.close()
        c["window"] = {"path": p, "start": s0 + a.offset, "end": s1 + a.offset}

json.dump(cands, sys.stdout, indent=1)
print(f"{len(cands)} candidates over {S / 3600:.1f} h", file=sys.stderr)
