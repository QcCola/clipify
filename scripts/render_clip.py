#!/usr/bin/env python3
"""Render one vertical clip in the stream format (see formats/gta6.md).

  render_clip.py --src trimmed.mp4 --out final.mp4 --layout cam_game \
      --cam 0,0,472,264 [--game 0,0,1920,1080] \
      --hook "😭 He got paid $12 to paint" --phrases phrases.json [--emph-check]

Layouts (canvas default 720x1280, 9:16):
  game      sharp gameplay panel (full wide shot) centred over a blurred,
            darkened duplicate; captions under the panel.
  cam_game  facecam panel on top, captions in the gap, gameplay panel below;
            blurred gameplay fills the canvas. Captions never cover face or action.
  cam_only  facecam panel (crop the ROI as tight/loose as the moment needs),
            blurred camera fills the canvas.

Hook: rendered with Pillow (Barlow Bold 58 px + Noto Color Emoji), white,
3 px charcoal outline, subtle shadow, max 2 lines inside the middle 80%,
pinned to the top for --hook-dur seconds (default: whole clip).

Captions: --phrases (guide style, steady phrases, warm-yellow emph word) or
--whisper + --style karaoke|opus|minimal (word highlight).

Output: H.264 + AAC MP4, source fps capped at 60, square pixels. The file is
probed after encoding and the script exits 1 if size/SAR/codecs are wrong.
"""
import argparse, json, os, subprocess, sys, tempfile
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.dirname(HERE)
FONT_DIR = os.path.join(SKILL, "fonts")
BARLOW = os.path.join(FONT_DIR, "Barlow-Bold.ttf")
EMOJI_FONTS = ["/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
               "/System/Library/Fonts/Apple Color Emoji.ttc"]
sys.path.insert(0, HERE)
from build_ass import build_guide_ass, build_word_ass, load_words  # noqa: E402

CAPTION_BLOCK = 130   # two lines of 50px Barlow + outline, with breathing room
GAP = 14


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,codec_name,width,height,sample_aspect_ratio,r_frame_rate:format=duration",
                          "-of", "json", path], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def roi(s):
    x, y, w, h = (int(v) for v in s.split(","))
    return x, y, w, h


def even(v):
    return int(v) // 2 * 2


# ---------------------------------------------------------------- hook text
def _is_emoji(ch):
    o = ord(ch)
    return (0x1F000 <= o <= 0x1FAFF or 0x2600 <= o <= 0x27BF or 0x2B00 <= o <= 0x2BFF
            or o in (0x200D, 0xFE0F) or 0x1F1E6 <= o <= 0x1F1FF)


def render_hook_png(text, out_png, canvas_w, size=58, max_lines=2):
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    emoji_path = next((p for p in EMOJI_FONTS if os.path.exists(p)), None)
    max_w = int(canvas_w * 0.80)

    def runs(s):  # split into (is_emoji, text) runs
        r = []
        for ch in s:
            e = _is_emoji(ch)
            if r and r[-1][0] == e: r[-1][1] += ch
            else: r.append([e, ch])
        return r

    def layout(sz):
        font = ImageFont.truetype(BARLOW, sz)
        efont = ImageFont.truetype(emoji_path, 109) if emoji_path else None
        esz = int(sz * 1.05)

        def width(s):
            w = 0
            for e, t in runs(s):
                if e:
                    w += sum(esz + 4 for c in t if c not in "‍️") if efont else 0
                else:
                    w += font.getlength(t)
            return w
        words, lines, cur = text.split(), [], ""
        for wd in words:
            cand = (cur + " " + wd).strip()
            if width(cand) <= max_w or not cur: cur = cand
            else: lines.append(cur); cur = wd
        if cur: lines.append(cur)
        return font, efont, esz, lines, width

    sz = size
    while True:
        font, efont, esz, lines, width = layout(sz)
        if len(lines) <= max_lines or sz <= 40: break
        sz -= 4
    lines = lines[:max_lines]
    line_h = int(sz * 1.22)
    pad = 12
    img = Image.new("RGBA", (canvas_w, line_h * len(lines) + pad * 2), (0, 0, 0, 0))
    txt = Image.new("RGBA", img.size, (0, 0, 0, 0))
    emo = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(txt)
    for i, line in enumerate(lines):
        x = (canvas_w - width(line)) / 2
        y = pad + i * line_h
        for e, t in runs(line):
            if e:
                if not efont: continue
                for c in t:
                    if c in "‍️": continue
                    g = Image.new("RGBA", (140, 140), (0, 0, 0, 0))
                    ImageDraw.Draw(g).text((0, 0), c, font=efont, embedded_color=True)
                    bb = g.getbbox()
                    if bb: g = g.crop(bb)
                    g = g.resize((esz, int(esz * g.height / max(1, g.width))), Image.LANCZOS)
                    emo.alpha_composite(g, (int(x) + 2, int(y + (sz - g.height) * 0.55)))
                    x += esz + 4
            else:
                d.text((x, y), t, font=font, fill="white", stroke_width=3, stroke_fill=(38, 38, 38, 255))
                x += font.getlength(t)
    # subtle shadow
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    a = txt.split()[3].point(lambda v: int(v * 0.45))
    shadow.putalpha(a)
    shadow = shadow.filter(ImageFilter.GaussianBlur(2))
    img.alpha_composite(shadow, (2, 3))
    img.alpha_composite(txt)
    img.alpha_composite(emo)
    img.save(out_png)
    return img.height


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--start", type=float, default=0, help="trim: seconds into --src (captions must be relative to this; see make_phrases.py --offset)")
    ap.add_argument("--end", type=float, default=None, help="trim: end time in --src seconds")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layout", choices=["game", "cam_game", "cam_only"], required=True)
    ap.add_argument("--cam", help="facecam ROI x,y,w,h in source pixels")
    ap.add_argument("--game", help="gameplay ROI x,y,w,h (default: full frame)")
    ap.add_argument("--cam-width", type=int, default=None, help="facecam panel width on canvas")
    ap.add_argument("--hook", default="")
    ap.add_argument("--hook-dur", type=float, default=None)
    ap.add_argument("--phrases"); ap.add_argument("--whisper"); ap.add_argument("--style", default="guide")
    ap.add_argument("--w", type=int, default=720); ap.add_argument("--h", type=int, default=1280)
    ap.add_argument("--max-fps", type=float, default=60)
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--keep-overlay", action="store_true",
                    help="cam_game: leave the stream's embedded facecam visible inside the gameplay panel "
                         "(default blurs it, since the camera already has its own panel)")
    ap.add_argument("--keep-tmp", action="store_true")
    a = ap.parse_args()

    W, H = a.w, a.h
    info = probe(a.src)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    has_audio = any(s["codec_type"] == "audio" for s in info["streams"])
    sw, sh = v["width"], v["height"]
    fps = float(Fraction(v["r_frame_rate"]))
    out_fps = min(fps, a.max_fps)
    game = roi(a.game) if a.game else (0, 0, sw, sh)
    cam = roi(a.cam) if a.cam else None
    if a.layout in ("cam_game", "cam_only") and not cam:
        sys.exit("--cam x,y,w,h is required for this layout")

    tmp = tempfile.mkdtemp(prefix="clipify_render_")
    # hook
    hook_h, hook_png = 0, None
    top_y = 40
    if a.hook.strip():
        hook_png = os.path.join(tmp, "hook.png")
        hook_h = render_hook_png(a.hook.strip(), hook_png, W)
    content_top = top_y + hook_h + 18
    text_floor = int(H * 0.80)          # captions must end above bottom 20%

    # panel geometry
    panels = []  # (name, roi, x, y, w, h)
    if a.layout == "game":
        gw = W; gh = even(gw * game[3] / game[2])
        stack = gh + GAP + CAPTION_BLOCK
        y0 = max(content_top, (H - stack) // 2)
        if y0 + stack > text_floor: y0 = max(content_top, text_floor - stack)
        panels.append(("game", game, 0, even(y0), gw, gh))
        cap_top = y0 + gh + GAP
    elif a.layout == "cam_game":
        gw = W; gh = even(gw * game[3] / game[2])
        cw = even(a.cam_width or int(W * 0.78)); ch = even(cw * cam[3] / cam[2])
        avail = H - 24 - content_top
        stack = ch + GAP + CAPTION_BLOCK + GAP + gh
        if stack > avail:              # shrink the cam panel until everything fits
            ch = even(max(160, ch - (stack - avail))); cw = even(ch * cam[2] / cam[3])
            stack = ch + GAP + CAPTION_BLOCK + GAP + gh
        y0 = content_top + max(0, (avail - stack) // 2)
        cap_top = y0 + ch + GAP
        if cap_top + CAPTION_BLOCK > text_floor:   # pull everything up
            y0 -= cap_top + CAPTION_BLOCK - text_floor; y0 = max(content_top, y0); cap_top = y0 + ch + GAP
        panels.append(("cam", cam, (W - cw) // 2, even(y0), cw, ch))
        panels.append(("game", game, 0, even(cap_top + CAPTION_BLOCK + GAP), gw, gh))
    else:  # cam_only
        cw = even(a.cam_width or W); ch = even(cw * cam[3] / cam[2])
        max_h = text_floor - GAP - CAPTION_BLOCK - content_top
        if ch > max_h: ch = even(max_h); cw = even(ch * cam[2] / cam[3])
        y0 = content_top + max(0, (text_floor - GAP - CAPTION_BLOCK - content_top - ch) // 2)
        panels.append(("cam", cam, (W - cw) // 2, even(y0), cw, ch))
        cap_top = y0 + ch + GAP
    cap_top = int(cap_top)

    # captions
    ass = None
    if a.phrases:
        ass = build_guide_ass(json.load(open(a.phrases)), os.path.join(tmp, "cap.ass"), W, H, top=cap_top)
    elif a.whisper:
        ass = build_word_ass(load_words(a.whisper), os.path.join(tmp, "cap.ass"),
                             a.style if a.style != "guide" else "karaoke", W, H, top=cap_top)

    # filter graph
    bg_roi = cam if a.layout == "cam_only" else game
    n = len(panels) + 1
    f = [f"[0:v]fps={out_fps:g},setsar=1,split={n}[bgsrc]" + "".join(f"[p{i}]" for i in range(len(panels)))]
    bx, by, bw, bh = bg_roi
    f.append(f"[bgsrc]crop={bw}:{bh}:{bx}:{by},scale={W//4}:{H//4}:force_original_aspect_ratio=increase,"
             f"crop={W//4}:{H//4},gblur=sigma=8,eq=brightness=-0.22:saturation=0.85,scale={W}:{H}[bg]")
    last = "bg"
    for i, (name, (x, y, w, h), px, py, pw, ph) in enumerate(panels):
        src = f"p{i}"
        if name == "game" and a.layout == "cam_game" and cam and not a.keep_overlay:
            cx, cy, cw_, ch_ = cam
            ix, iy = max(cx, x), max(cy, y)
            iw, ih = min(cx + cw_, x + w) - ix, min(cy + ch_, y + h) - iy
            if iw > 8 and ih > 8:   # embedded cam overlaps the gameplay crop: blur it out
                f.append(f"[p{i}]split=2[gm{i}][gb{i}];[gb{i}]crop={even(iw)}:{even(ih)}:{ix}:{iy},gblur=sigma=25,"
                         f"eq=brightness=-0.1[gbb{i}];[gm{i}][gbb{i}]overlay={ix}:{iy}[gx{i}]")
                src = f"gx{i}"
        f.append(f"[{src}]crop={w}:{h}:{x}:{y},scale={pw}:{ph}:flags=lanczos[pp{i}]")
        f.append(f"[{last}][pp{i}]overlay={px}:{py}[l{i}]")
        last = f"l{i}"
    inputs = (["-ss", f"{a.start:.3f}"] if a.start else []) + (["-to", f"{a.end:.3f}"] if a.end else []) + ["-i", a.src]
    if hook_png:
        inputs += ["-loop", "1", "-i", hook_png]
        en = f":enable='lt(t,{a.hook_dur})'" if a.hook_dur else ""
        f.append(f"[{last}][1:v]overlay=0:{top_y}:shortest=1{en}[lh]")
        last = "lh"
    if ass:
        esc = ass.replace(":", "\\:")
        f.append(f"[{last}]subtitles={esc}:fontsdir={FONT_DIR}[ls]")
        last = "ls"
    f.append(f"[{last}]format=yuv420p,setsar=1[vout]")

    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *inputs, "-filter_complex", ";".join(f),
           "-map", "[vout]"]
    if has_audio: cmd += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "160k"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(a.crf), "-r", f"{out_fps:g}",
            "-movflags", "+faststart", "-shortest", a.out]
    subprocess.run(cmd, check=True)

    # verify
    o = probe(a.out)
    ov = next(s for s in o["streams"] if s["codec_type"] == "video")
    oa = [s for s in o["streams"] if s["codec_type"] == "audio"]
    problems = []
    if (ov["width"], ov["height"]) != (W, H): problems.append(f"size {ov['width']}x{ov['height']}")
    if ov.get("sample_aspect_ratio", "1:1") not in ("1:1", "N/A", "0:1"): problems.append(f"SAR {ov['sample_aspect_ratio']}")
    if ov["codec_name"] != "h264": problems.append(f"vcodec {ov['codec_name']}")
    if has_audio and (not oa or oa[0]["codec_name"] != "aac"): problems.append("audio not AAC")
    if float(Fraction(ov["r_frame_rate"])) > 60.01: problems.append(f"fps {ov['r_frame_rate']}")
    report = {"out": a.out, "size": f"{ov['width']}x{ov['height']}", "sar": ov.get("sample_aspect_ratio"),
              "fps": ov["r_frame_rate"], "duration": round(float(o["format"]["duration"]), 2),
              "layout": a.layout, "caption_top": cap_top, "ok": not problems, "problems": problems}
    print(json.dumps(report))
    if not a.keep_tmp:
        import shutil; shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
