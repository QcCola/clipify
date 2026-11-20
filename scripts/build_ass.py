#!/usr/bin/env python3
"""Build ASS captions.

Two kinds of input:
  * word presets (opus / karaoke / minimal): whisper JSON with word timestamps,
    word-by-word highlight.
  * guide preset (default for stream clips): a phrases JSON from make_phrases.py,
    steady phrase captions in Barlow Bold, sentence case, optional single
    warm-yellow emphasis word per phrase (the phrase's "emph" field).

CLI:
  build_ass.py WHISPER.json OUT.ass [opus|karaoke|minimal]      # legacy, 1080x1920
  build_ass.py --phrases PHRASES.json OUT.ass [--top Y] [--w 720 --h 1280]

render_clip.py imports build_guide_ass / build_word_ass directly so it can place
captions in the gap its layout leaves free. Burn with
  subtitles=FILE.ass:fontsdir=<skill-dir>/fonts
so libass finds the bundled Barlow Bold.
"""
import json, sys, argparse

WHITE = "&H00FFFFFF&"
CHARCOAL = "&H00262626&"          # 3px charcoal outline
SHADOW = "&H80000000&"            # subtle 50% shadow
WARM_YELLOW = "&H0040C8FF&"       # #FFC840 in ASS BGR order

WORD_PRESETS = {
    "opus":    dict(font="Barlow", size=100, chunk=3, highlight="&H0000FFFF&", outline=8, shadow=3),
    "karaoke": dict(font="Barlow", size=110, chunk=4, highlight="&H0000FF00&", outline=6, shadow=2),
    "minimal": dict(font="Barlow", size=70,  chunk=6, highlight=None,          outline=4, shadow=1),
}


def fmt_time(t):
    t = max(0.0, t)
    h = int(t // 3600); m = int((t % 3600) // 60); s = t - h * 3600 - m * 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _header(w, h, style_line):
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{style_line}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _esc(s):
    return s.replace("{", "(").replace("}", ")")


def build_guide_ass(phrases, out, w=720, h=1280, top=None, size=50, outline=3, shadow=1.5):
    """Steady phrase captions. `top` = y of the caption block's top edge
    (alignment 8, top-centre). Default: bottom edge sits at 80% of height."""
    margin_lr = int(w * 0.10)                     # keep inside the middle 80%
    if top is None:
        align, margin_v = 2, int(h * 0.20)        # bottom-centre, above bottom 20%
    else:
        align, margin_v = 8, int(top)
    style = (f"Style: Default,Barlow,{size},{WHITE},&H000000FF,{CHARCOAL},{SHADOW},"
             f"-1,0,0,0,100,100,0,0,1,{outline},{shadow},{align},{margin_lr},{margin_lr},{margin_v},1")
    ev = []
    for p in phrases:
        text = _esc(p["text"])
        emph = (p.get("emph") or "").strip()
        if emph:
            import re
            # match the emphasis even if make_phrases put a line break inside it
            pat = r"\s+".join(re.escape(t) for t in emph.split())
            m = re.search(pat, text)
            if m:
                seg = m.group(0).replace("\n", "{\\c" + WHITE + "}\n{\\c" + WARM_YELLOW + "}")
                text = text[:m.start()] + "{\\c" + WARM_YELLOW + "}" + seg + "{\\c" + WHITE + "}" + text[m.end():]
        text = text.replace("\n", "\\N")
        ev.append(f"Dialogue: 0,{fmt_time(p['start'])},{fmt_time(p['end'])},Default,,0,0,0,,{text}")
    open(out, "w").write(_header(w, h, style) + "\n".join(ev) + "\n")
    return out


def load_words(whisper_json):
    data = json.load(open(whisper_json))
    return [{"start": w["start"], "end": w["end"], "text": w["word"].strip()}
            for seg in data["segments"] for w in seg.get("words", []) if w["word"].strip()]


def build_word_ass(words, out, preset="opus", w=1080, h=1920, top=None):
    """Word-by-word highlight presets. Sizes are defined for 1080 wide and scale with w."""
    P = dict(WORD_PRESETS.get(preset, WORD_PRESETS["opus"]))
    k = w / 1080
    size, outl, shad = int(P["size"] * k), max(1, round(P["outline"] * k)), max(0, round(P["shadow"] * k))
    margin_lr = int(w * 0.10)
    align, margin_v = (2, int(h * 0.20)) if top is None else (8, int(top))
    style = (f"Style: Default,{P['font']},{size},{WHITE},&H000000FF,&H00000000,&H00000000,"
             f"-1,0,0,0,100,100,0,0,1,{outl},{shad},{align},{margin_lr},{margin_lr},{margin_v},1")
    chunks = [words[i:i + P["chunk"]] for i in range(0, len(words), P["chunk"])]
    ev = []
    for chunk in chunks:
        for i, wd in enumerate(chunk):
            s = wd["start"]
            e = chunk[i + 1]["start"] if i + 1 < len(chunk) else chunk[-1]["end"]
            if e <= s: e = s + 0.05
            if P["highlight"]:
                line = " ".join((f"{{\\c{P['highlight']}}}{_esc(x['text'])}{{\\c{WHITE}}}" if j == i else _esc(x["text"]))
                                for j, x in enumerate(chunk))
            else:
                line = " ".join(_esc(x["text"]) for x in chunk)
            ev.append(f"Dialogue: 0,{fmt_time(s)},{fmt_time(e)},Default,,0,0,0,,{line}")
    open(out, "w").write(_header(w, h, style) + "\n".join(ev) + "\n")
    return out


if __name__ == "__main__":
    if "--phrases" in sys.argv:
        ap = argparse.ArgumentParser()
        ap.add_argument("--phrases", required=True); ap.add_argument("out")
        ap.add_argument("--top", type=int); ap.add_argument("--w", type=int, default=720); ap.add_argument("--h", type=int, default=1280)
        a = ap.parse_args()
        build_guide_ass(json.load(open(a.phrases)), a.out, a.w, a.h, a.top)
        print(f"wrote {a.out}", file=sys.stderr)
    else:
        words = load_words(sys.argv[1])
        build_word_ass(words, sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "opus")
        print(f"wrote {sys.argv[2]}: {len(words)} words", file=sys.stderr)
