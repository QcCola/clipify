#!/usr/bin/env python3
"""Render one vertical clip in the stream format (see formats/gta6.md).

  render_clip.py --src trimmed.mp4 --out final.mp4 --layout cam_game \
      --cam 0,0,472,264 [--game 0,0,1920,1080] \
      --hook "😭 He got paid $12 to paint" --phrases phrases.json [--emph-check]

Layouts (canvas default 720x1280, 9:16):
  stack     (default for facecam streams) full-bleed split, no gaps or blur:
            facecam fills the top --split of the canvas (default 30%), zoomed
            in on the face (--cam-zoom, --cam-focus); gameplay fills the rest,
            cropped from the centre (--game-focus, --game-zoom) just enough to
            fit. Hook sits on the seam; captions in the lower gameplay, above
            the bottom 20%. The gameplay crop loses the frame's sides: when the
            payoff spans the full width (cars from opposite edges), use cam_game.
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

Length: --max-dur (default 30 s) is a hard cap. A longer trim is refused
before encoding; tighten --start/--end instead (or pass --max-dur 0 to lift it).

Output: H.264 + AAC MP4, source fps capped at 60, square pixels. The file is
probed after encoding and the script exits 1 if size/SAR/codecs/length are wrong.
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
    ap.add_argument("--layout", choices=["cam_game", "stack", "game", "cam_only"], default="cam_game")
    ap.add_argument("--wide", action="store_true", help="cam_game: show the whole 16:9 gameplay shot (letterboxed) "
                    "instead of filling the canvas; use when the payoff spans the full width")
    ap.add_argument("--split", type=float, default=0.30, help="stack: fraction of canvas height for the facecam")
    ap.add_argument("--cam-zoom", type=float, default=1.15, help="stack: zoom into the cam beyond fill (1 = just fill)")
    ap.add_argument("--cam-focus", help="stack: cx,cy in source px to centre the cam crop on (default: ROI centre)")
    ap.add_argument("--game-zoom", type=float, default=1.0, help="cam_game/stack: zoom into gameplay beyond fill")
    ap.add_argument("--game-focus", help="cam_game/stack: cx,cy in source px to centre the gameplay crop on (default: centre)")
    ap.add_argument("--hook-pos", choices=["auto", "top", "seam"], default="auto")
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
    ap.add_argument("--max-dur", type=float, default=30, help="hard cap on clip length in seconds (0 = no cap)")
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
    if a.layout in ("stack", "cam_game", "cam_only") and not cam:
        sys.exit("--cam x,y,w,h is required for this layout")
    want = (a.end if a.end else float(info["format"]["duration"])) - a.start
    if a.max_dur and want > a.max_dur + 0.05:
        sys.exit(f"clip would run {want:.1f} s, over the {a.max_dur:g} s cap: tighten --start/--end "
                 f"(open closer to the payoff, end on the reaction)")

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
    panels = []  # (name, source crop x,y,w,h, canvas x, y, w, h)

    def fill_crop(r, aspect, zoom, focus):
        """Largest aspect-correct box inside ROI r, shrunk by zoom, centred on focus, clamped to r."""
        rx, ry, rw, rh = r
        if rw / rh > aspect: ch_, cw_ = rh, rh * aspect
        else: cw_, ch_ = rw, rw / aspect
        cw_, ch_ = cw_ / zoom, ch_ / zoom
        fx, fy = focus if focus else (rx + rw / 2, ry + rh / 2)
        x0 = min(max(rx, fx - cw_ / 2), rx + rw - cw_); y0 = min(max(ry, fy - ch_ / 2), ry + rh - ch_)
        return (even(x0), even(y0), even(cw_), even(ch_))

    def pt(sv):
        return tuple(float(v) for v in sv.split(",")) if sv else None

    if a.layout == "stack":
        ch = even(H * a.split); gh = H - ch
        panels.append(("cam", fill_crop(cam, W / ch, a.cam_zoom, pt(a.cam_focus)), 0, 0, W, ch))
        panels.append(("game", fill_crop(game, W / gh, a.game_zoom, pt(a.game_focus)), 0, ch, W, gh))
        cap_top = None                  # bottom-anchored: block ends at 80% height
    elif a.layout == "game":
        gw = W; gh = even(gw * game[3] / game[2])
        stack = gh + GAP + CAPTION_BLOCK
        y0 = max(content_top, (H - stack) // 2)
        if y0 + stack > text_floor: y0 = max(content_top, text_floor - stack)
        panels.append(("game", game, 0, even(y0), gw, gh))
        cap_top = y0 + gh + GAP
    elif a.layout == "cam_game":
        # hook on top, facecam panel, captions in the gap, then gameplay filling the rest of
        # the canvas to the bottom edge: cropped in from the sides just enough to fit
        # (--game-zoom / --game-focus); --wide keeps the whole 16:9 shot letterboxed instead
        cw = even(a.cam_width or int(W * 0.78)); ch = even(cw * cam[3] / cam[2])
        y0 = content_top
        cap_top = y0 + ch + GAP
        if cap_top + CAPTION_BLOCK > text_floor:
            ch = even(text_floor - CAPTION_BLOCK - GAP - y0); cw = even(ch * cam[2] / cam[3]); cap_top = y0 + ch + GAP
        panels.append(("cam", cam, (W - cw) // 2, even(y0), cw, ch))
        gy = even(cap_top + CAPTION_BLOCK + GAP)
        if a.wide:
            gh = even(W * game[3] / game[2])
            panels.append(("game", game, 0, gy + max(0, (H - gy - gh) // 2), W, gh))
        else:
            gh = H - gy
            gc = fill_crop(game, W / gh, a.game_zoom, pt(a.game_focus))
            # slide the crop sideways off the stream's embedded facecam if a small shift does it
            # (cleaner than blurring a patch over it)
            gx0, gy0, gw0, gh0 = gc
            ox, oy, ow, oh = cam
            if ox < gx0 + gw0 and ox + ow > gx0:
                gx_min, gx_max = game[0], game[0] + game[2] - gw0
                right = ox + ow                     # overlay on the left: start after it
                left = ox - gw0                     # overlay on the right: end before it
                for nx in sorted([right, left], key=lambda v: abs(v - gx0)):
                    if gx_min <= nx <= gx_max and abs(nx - gx0) <= 0.15 * game[2]:
                        gc = (even(nx + 1), gy0, gw0, gh0); break
            panels.append(("game", gc, 0, gy, W, gh))
    else:  # cam_only
        cw = even(a.cam_width or W); ch = even(cw * cam[3] / cam[2])
        max_h = text_floor - GAP - CAPTION_BLOCK - content_top
        if ch > max_h: ch = even(max_h); cw = even(ch * cam[2] / cam[3])
        y0 = content_top + max(0, (text_floor - GAP - CAPTION_BLOCK - content_top - ch) // 2)
        panels.append(("cam", cam, (W - cw) // 2, even(y0), cw, ch))
        cap_top = y0 + ch + GAP
    cap_top = int(cap_top) if cap_top is not None else None

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
        if name == "game" and a.layout in ("cam_game", "stack") and cam and not a.keep_overlay:
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
        hook_y = top_y
        if a.hook_pos == "seam" or (a.hook_pos == "auto" and a.layout == "stack"):
            hook_y = max(0, panels[0][5] - hook_h // 2)     # straddle the cam/gameplay edge
        f.append(f"[{last}][1:v]overlay=0:{hook_y}:shortest=1{en}[lh]")
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
    if a.max_dur and float(o["format"]["duration"]) > a.max_dur + 0.25: problems.append(f"duration {o['format']['duration']} > {a.max_dur:g}")
    report = {"out": a.out, "size": f"{ov['width']}x{ov['height']}", "sar": ov.get("sample_aspect_ratio"),
              "fps": ov["r_frame_rate"], "duration": round(float(o["format"]["duration"]), 2),
              "layout": a.layout, "caption_top": cap_top, "ok": not problems, "problems": problems}
    print(json.dumps(report))
    if not a.keep_tmp:
        import shutil; shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
