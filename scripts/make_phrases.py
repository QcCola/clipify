#!/usr/bin/env python3
"""Turn whisper word timestamps into editable caption phrases.

Usage: make_phrases.py WHISPER.json [--offset S] [--end S] [--max-line 24] > phrases.json

Each phrase: {"start", "end", "text", "emph"}.
  * starts exactly when its first word is spoken (punchline appears on delivery)
  * breaks at sentence punctuation, pauses > 0.35 s, or when it would exceed
    two lines; a long phrase gets one \\n at the most natural point near the middle
  * sentence case: first letter capitalised, SHOUTED words lowered unless they
    look like acronyms (PS5, GTA, PC)
  * "emph" is empty: fill in ONE word per phrase only where emphasis helps the
    line land (rendered warm yellow). Fix misheard names/words in "text" by ear.
--offset shifts all times (use when you trimmed the head off the clip).
"""
import json, sys, argparse, re

ap = argparse.ArgumentParser()
ap.add_argument("whisper_json")
ap.add_argument("--offset", type=float, default=0.0, help="seconds to subtract from every timestamp")
ap.add_argument("--end", type=float, default=None, help="drop words after this (clip-relative, after offset)")
ap.add_argument("--max-line", type=int, default=24, help="chars per line (Barlow Bold 50px in 576px ≈ 24)")
a = ap.parse_args()

words = []
for seg in json.load(open(a.whisper_json))["segments"]:
    for w in seg.get("words", []):
        t = w["word"].strip()
        if not t: continue
        s, e = w["start"] - a.offset, w["end"] - a.offset
        if s < 0 or (a.end is not None and s > a.end): continue
        words.append({"s": s, "e": e, "t": t})

ACRONYM = re.compile(r"^[A-Z0-9]{2,4}[!?.,]*$")
def case(t, first):
    core = re.sub(r"[^A-Za-z]", "", t)
    if core.isupper() and len(core) > 1 and not ACRONYM.match(t) and core != "I":
        t = t.lower()
    if first and t[:1].islower():
        t = t[:1].upper() + t[1:]
    return t

MAX = a.max_line * 2
DANGLING = {"of", "to", "the", "a", "an", "and", "for", "with", "in", "on", "my", "your", "at", "from", "but", "or", "so", "is", "was"}
phrases, cur = [], []
for i, w in enumerate(words):
    cur.append(w)
    text_len = len(" ".join(x["t"] for x in cur))
    nxt = words[i + 1] if i + 1 < len(words) else None
    gap = (nxt["s"] - w["e"]) if nxt else 9
    end_sentence = w["t"][-1:] in ".?!"
    soft = w["t"][-1:] in ",;:" and len(cur) >= 3
    too_long = nxt is not None and text_len + 1 + len(nxt["t"]) > MAX
    if end_sentence or soft or gap > 0.35 or too_long or nxt is None:
        carry = []
        # never end a forced (too-long) break on a dangling function word: "five minutes of / work."
        while too_long and not (end_sentence or soft or gap > 0.35) and len(cur) > 2 \
                and cur[-1]["t"].lower().strip(",") in DANGLING:
            carry.insert(0, cur.pop())
        phrases.append(cur); cur = carry

# merge orphans: a lone word ("too.", "outside.", "Like,") joins its neighbour when it fits
merged = []
for ph in phrases:
    if merged and len(ph) == 1 and len(" ".join(x["t"] for x in merged[-1] + ph)) <= MAX + 8 \
            and ph[0]["s"] - merged[-1][-1]["e"] < 0.8 and merged[-1][-1]["t"][-1:] not in ".?!":
        merged[-1] = merged[-1] + ph
    elif merged and len(merged[-1]) == 1 and merged[-1][0]["t"][-1:] not in ".?!" \
            and len(" ".join(x["t"] for x in merged[-1] + ph)) <= MAX and ph[0]["s"] - merged[-1][0]["e"] < 1.2:
        merged[-1] = merged[-1] + ph
    else:
        merged.append(ph)
phrases = merged

SPLIT_BEFORE = {"and", "but", "so", "because", "cause", "or", "then", "when", "if", "like", "that", "to", "for", "with"}
def two_lines(toks):
    s = " ".join(toks)
    if len(s) <= a.max_line: return s
    best, bscore = None, 1e9
    for k in range(1, len(toks)):
        l1, l2 = " ".join(toks[:k]), " ".join(toks[k:])
        if len(l1) > a.max_line + 4 or len(l2) > a.max_line + 4: continue
        score = abs(len(l1) - len(l2))
        if toks[k - 1][-1:] in ",.;:!?": score -= 8
        if toks[k].lower().strip(",.") in SPLIT_BEFORE: score -= 4
        if score < bscore: best, bscore = k, score
    if best is None: best = len(toks) // 2
    return " ".join(toks[:best]) + "\n" + " ".join(toks[best:])

out = []
for j, ph in enumerate(phrases):
    first_of_sentence = j == 0 or phrases[j - 1][-1]["t"][-1:] in ".?!"
    toks = [case(w["t"], first=(k == 0 and first_of_sentence)) for k, w in enumerate(ph)]
    start = ph[0]["s"]
    nxt_start = phrases[j + 1][0]["s"] if j + 1 < len(phrases) else None
    end = ph[-1]["e"] + 0.4
    if nxt_start is not None: end = min(end if nxt_start - ph[-1]["e"] > 1.0 else nxt_start, nxt_start)
    out.append({"start": round(start, 2), "end": round(max(end, start + 0.5), 2), "text": two_lines(toks), "emph": ""})

json.dump(out, sys.stdout, indent=1, ensure_ascii=False)
print(f"{len(out)} phrases", file=sys.stderr)
