#!/usr/bin/env python3
"""Transcribe every scout window in one process and write a reading digest.

  transcribe.py candidates.json OUTDIR [--model tiny.en] > digest.md

Loads whisper once (model reload per file is most of the cost on CPU), writes
OUTDIR/w_<t>.json (word timestamps, window-relative) for each candidate, and
prints a markdown digest: one block per candidate with ABSOLUTE stream
timestamps per line, so you can read for setup -> punchline -> reaction and
cut straight from the times shown.
"""
import argparse, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("candidates"); ap.add_argument("outdir")
ap.add_argument("--model", default="tiny.en")
a = ap.parse_args()

import whisper  # noqa: E402
model = whisper.load_model(a.model)
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
        r = model.transcribe(win["path"], language="en", word_timestamps=True,
                             condition_on_previous_text=False, fp16=False)
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
