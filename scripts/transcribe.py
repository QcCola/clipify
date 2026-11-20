#!/usr/bin/env python3
"""Transcribe with word timestamps: all scout windows at once, or one clip.

  transcribe.py candidates.json OUTDIR [--model small.en] > digest.md
  transcribe.py --file clip.wav --out clip.json [--model small.en]

Engines (--engine auto):
  * windows mode -> faster-whisper (CTranslate2, int8 CPU): small.en ~9x real
    time on 4 CPUs (tiny.en ~25x), so the accurate model is affordable for every
    window. Falls back to openai-whisper if not installed / model download blocked.
    Models download once from Hugging Face: huggingface.co, cas-server.xethub.hf.co,
    cas-bridge.xethub.hf.co, transfer.xethub.hf.co, us.aws.cdn.hf.co.
  * --file mode -> openai-whisper: on short caption clips it keeps punctuation
    (make_phrases breaks on it) and the final line, which faster-whisper didn't.

Input must be 16 kHz mono 16-bit WAV (fetch.py audio / scout.py --windows
write that; for a clip: ffmpeg -nostdin -i clip.mp4 -vn -ac 1 -ar 16000 clip.wav).
Audio is passed as samples, not a path, which sidesteps PyAV version breakage.

Output JSON matches openai-whisper's shape ({"text", "segments": [{"start",
"end", "text", "words": [{"word", "start", "end"}]}]}) so make_phrases.py and
build_ass.py read either engine's output.

Windows mode prints a markdown digest: one block per candidate with ABSOLUTE
stream timestamps per line (<< marks the scout peak), to read for
setup -> punchline -> reaction and cut straight from the times shown.
"""
import argparse, json, os, sys, wave
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("candidates", nargs="?"); ap.add_argument("outdir", nargs="?")
ap.add_argument("--file"); ap.add_argument("--out")
ap.add_argument("--model", default="small.en")
ap.add_argument("--engine", choices=["auto", "faster", "openai"], default="auto")
ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
a = ap.parse_args()


def read_wav(p):
    w = wave.open(p)
    assert w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == 16000, \
        f"{p}: need 16 kHz mono 16-bit wav"
    return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def load_engine():
    # auto: faster-whisper for scout windows (speed), openai-whisper for a single clip
    # (--file): on caption clips openai-whisper keeps punctuation (phrases break on it)
    # and the final line; faster-whisper either drops punctuation or the last line.
    engine = a.engine
    if engine == "auto":
        engine = "openai" if a.file else "faster"
    if engine == "faster":
        try:
            from faster_whisper import WhisperModel
            m = WhisperModel(a.model, device="cpu", compute_type="int8", cpu_threads=a.threads)

            def run(path):
                x = np.concatenate([read_wav(path), np.zeros(16000 * 2, np.float32)])  # tail pad
                segs, _ = m.transcribe(x, language="en", word_timestamps=True,
                                       condition_on_previous_text=False, vad_filter=False,
                                       initial_prompt="Hey, what's up? Yeah, that's crazy, bro.")
                out = []
                for s in segs:
                    out.append({"start": s.start, "end": s.end, "text": s.text,
                                "words": [{"word": w.word, "start": w.start, "end": w.end} for w in (s.words or [])]})
                return {"text": "".join(s["text"] for s in out), "segments": out}
            print(f"engine: faster-whisper {a.model}", file=sys.stderr)
            return run
        except Exception as e:  # not installed / model download blocked
            if a.engine == "faster": raise
            print(f"faster-whisper unavailable ({type(e).__name__}: {str(e)[:120]}); using openai-whisper",
                  file=sys.stderr)
    import whisper
    m = whisper.load_model(a.model)
    print(f"engine: openai-whisper {a.model}", file=sys.stderr)
    return lambda path: m.transcribe(read_wav(path), language="en", word_timestamps=True,
                                     condition_on_previous_text=False, fp16=False)


run = load_engine()

if a.file:
    r = run(a.file)
    json.dump(r, open(a.out or os.path.splitext(a.file)[0] + ".json", "w"))
    print(f"wrote {a.out or os.path.splitext(a.file)[0] + '.json'}", file=sys.stderr)
    sys.exit(0)

if not (a.candidates and a.outdir):
    ap.error("give candidates.json OUTDIR, or --file")
cands = json.load(open(a.candidates))
os.makedirs(a.outdir, exist_ok=True)


def hms(t):
    t = int(t); return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}"


print(f"# Scout digest ({len(cands)} candidates)\n")
for i, c in enumerate(sorted(cands, key=lambda c: c["t"])):
    win = c.get("window")
    if not win: continue
    out = os.path.join(a.outdir, os.path.basename(win["path"]).replace(".wav", ".json"))
    if os.path.exists(out):
        r = json.load(open(out))
    else:
        r = run(win["path"])
        json.dump(r, open(out, "w"))
    print(f"## #{c['rank']} {c['hms']}  kind={c['kind']} score={c['score']}  "
          f"window {hms(win['start'])}-{hms(win['end'])}  json={out}")
    print(f"signals: {c['signals']}")
    for s in r["segments"]:
        txt = s["text"].strip()
        if txt:
            mark = " <<" if win["start"] + s["start"] <= c["t"] <= win["start"] + s["end"] + 1 else ""
            print(f"  {hms(win['start'] + s['start'])}  {txt}{mark}")
    print()
    print(f"\rtranscribed {i + 1}/{len(cands)}", end="", file=sys.stderr)
print(file=sys.stderr)
