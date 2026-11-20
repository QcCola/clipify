#!/usr/bin/env python3
"""Fetch from a stream/VOD URL (Twitch, YouTube, anything yt-dlp reads) without
downloading the whole video.

  fetch.py info   URL
  fetch.py audio  URL OUT.wav [--start S] [--end E] [--parts 4]
  fetch.py clip   URL START DUR OUT.mp4 [--format 1080p60/best]
  fetch.py frames URL OUTDIR T1 [T2 ...] [--sheet sheet.jpg]

* Behind an HTTPS proxy (cloud sandboxes) ffmpeg ignores HTTPS_PROXY; this
  passes it explicitly with -http_proxy.
* Every ffmpeg call gets -nostdin (an ffmpeg inside a shell `while read` loop
  otherwise eats the loop's input and silently processes the wrong times).
* audio: 16 kHz mono wav, downloaded as N parallel slices then joined.
  Twitch audio_only downloads at roughly 40x real time per slice.
* clip: frame-accurate cut, re-encoded at high quality (crf 16) for editing.
* frames: one still per timestamp (parallel), optional labelled contact sheet;
  use it to check what is on screen and what the layout is before cutting.
Prints the media host so a blocked-host error can be fixed by allowing it
(Twitch VODs come from a per-VOD CloudFront host).
YouTube may demand sign-in from datacenter IPs: pass --cookies cookies.txt.
"""
import argparse, json, os, subprocess, sys, shutil, tempfile
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

PROXY = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")


def ff(*args):
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y"]
    return cmd + (["-http_proxy", PROXY] if PROXY else []) + list(args)


def ytdlp(url, *args, cookies=None):
    cmd = ["yt-dlp", "--no-warnings"] + (["--cookies", cookies] if cookies else []) + list(args) + [url]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"yt-dlp failed:\n{r.stderr.strip()[-1500:]}")
    return r.stdout.strip()


def media_url(url, fmt, cookies=None):
    u = ytdlp(url, "-g", "-f", fmt, cookies=cookies).splitlines()[-1]
    print(f"media host: {urlparse(u).hostname}", file=sys.stderr)
    return u


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["info", "audio", "clip", "frames"])
    ap.add_argument("url"); ap.add_argument("rest", nargs="*")
    ap.add_argument("--start", type=float, default=0); ap.add_argument("--end", type=float)
    ap.add_argument("--parts", type=int, default=4)
    ap.add_argument("--format", default="1080p60/1080p/best[height<=1080]/best")
    ap.add_argument("--audio-format", default="Audio_Only/bestaudio/worst")
    ap.add_argument("--sheet"); ap.add_argument("--cookies")
    a = ap.parse_args()

    if a.cmd == "info":
        j = json.loads(ytdlp(a.url, "-J", cookies=a.cookies))
        print(json.dumps({k: j.get(k) for k in ("title", "uploader", "duration", "duration_string",
                                                "width", "height", "fps", "webpage_url", "upload_date")}, indent=1))
        return

    if a.cmd == "audio":
        out = a.rest[0]
        u = media_url(a.url, a.audio_format, a.cookies)
        end = a.end
        if end is None:
            end = float(json.loads(ytdlp(a.url, "-J", cookies=a.cookies))["duration"])
        total = end - a.start
        n = max(1, a.parts)
        step = total / n
        tmp = tempfile.mkdtemp(prefix="clipify_audio_")
        parts = [os.path.join(tmp, f"p{i}.wav") for i in range(n)]

        def get(i):
            subprocess.run(ff("-ss", f"{a.start + i * step:.3f}", "-t", f"{step:.3f}", "-i", u,
                              "-vn", "-ac", "1", "-ar", "16000", parts[i]), check=True)
        with ThreadPoolExecutor(n) as ex: list(ex.map(get, range(n)))
        lst = os.path.join(tmp, "list.txt")
        open(lst, "w").write("".join(f"file '{p}'\n" for p in parts))
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                        "-i", lst, "-c", "copy", out], check=True)
        shutil.rmtree(tmp)
        print(f"wrote {out}: {total / 3600:.2f} h from {a.start:.0f}s", file=sys.stderr)
        return

    if a.cmd == "clip":
        start, dur, out = float(a.rest[0]), float(a.rest[1]), a.rest[2]
        u = media_url(a.url, a.format, a.cookies)
        subprocess.run(ff("-ss", f"{start:.3f}", "-i", u, "-t", f"{dur:.3f}", "-c:v", "libx264", "-preset", "veryfast",
                          "-crf", "16", "-c:a", "aac", "-b:a", "192k", out), check=True)
        print(f"wrote {out}", file=sys.stderr)
        return

    if a.cmd == "frames":
        outdir, times = a.rest[0], [float(t) for t in a.rest[1:]]
        os.makedirs(outdir, exist_ok=True)
        u = media_url(a.url, a.format, a.cookies)

        def hms(t):
            t = int(t); return f"{t // 3600}\\:{t % 3600 // 60:02d}\\:{t % 60:02d}"

        def grab(t):
            p = os.path.join(outdir, f"f_{int(t)}.jpg")
            subprocess.run(ff("-ss", f"{t:.2f}", "-i", u, "-frames:v", "1", "-vf",
                              f"scale=480:-2,drawtext=text='{hms(t)}':x=8:y=8:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6",
                              p), check=True)
            return p
        with ThreadPoolExecutor(6) as ex: paths = list(ex.map(grab, times))
        print("\n".join(paths))
        if a.sheet and paths:
            cols = min(4, len(paths)); rows = (len(paths) + cols - 1) // cols
            ins = sum((["-i", p] for p in paths), [])
            pads = "".join(f"[{i}:v]" for i in range(len(paths)))
            blanks = cols * rows - len(paths)
            if blanks:
                ins += ["-f", "lavfi", "-i", f"color=black:s=480x270:d=1"] * blanks
                pads += "".join(f"[{len(paths) + i}:v]" for i in range(blanks))
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *ins, "-filter_complex",
                            f"{pads}xstack=inputs={cols * rows}:layout=" +
                            "|".join(f"{(i % cols) * 480}_{(i // cols) * 270}" for i in range(cols * rows)),
                            "-frames:v", "1", a.sheet], check=True)
            print(f"sheet: {a.sheet}", file=sys.stderr)


if __name__ == "__main__":
    main()
