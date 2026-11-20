#!/usr/bin/env python3
"""Make every rendered clip post-ready: caption, hashtags, title and source
credit per platform, checked against each platform's limits.

  post_kit.py clips.json [--viral viral.json] [--expect 10] [--max-dur 30]

clips.json (you write it after the newcomer review; paths are relative to it):
  {
    "creator": "Clix",                       # credited in every post
    "source_url": "https://www.twitch.tv/videos/2933739024",
    "platform": "Twitch",                    # where the stream lives (credit line)
    "base_tags": ["gta6", "clix"],           # the format guide's evergreen tags
    "clips": [
      {"rank": 1, "file": "queue/01_show_up_in_style.mp4",
       "title": "He said he drives better on his phone",   # short, spoiler-free
       "caption": "chat said he drives better while texting so he proved it 💀",
       "tags": ["texting", "badDriver", "gta6funny"],      # moment-specific, relatable
       "source_time": "3:37:01", "kind": "game", "why": "..."},
      ...
    ]
  }

Hashtags per post = the clip's own tags, then base_tags, de-duplicated,
lower-cased, 4-8 in total. Pick the clip's own tags yourself, per moment:
relevance can't be automated (a creator's top tag may be another game).
--viral lists viral.json's pool (tags that several real uploads of this
creator/stream used, ranked by views) at the end of posts.md to choose from.

Writes, next to the clips:
  <clip>.txt   one file per clip with ready-to-paste blocks for TikTok,
               Instagram Reels, YouTube Shorts (title + description) and X
  posts.md     the ranked list (title, source time, length, why, credit) and
               every post block, for review in one place
  posts.json   the same, machine-readable

Checks (exit 1 with a list of problems, nothing is half-written silently):
exactly --expect clips ranked 1..N, each file exists and runs <= --max-dur s,
caption 1-150 chars, YouTube title <= 100 chars with #shorts, X post <= 280,
4-8 hashtags of [a-z0-9_] only.
"""
import argparse, json, os, re, subprocess, sys

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("clips")
ap.add_argument("--viral")
ap.add_argument("--expect", type=int, default=10, help="clips the batch must contain (0 = any)")
ap.add_argument("--max-dur", type=float, default=30)
a = ap.parse_args()

base = os.path.dirname(os.path.abspath(a.clips))
spec = json.load(open(a.clips))
creator = spec.get("creator", "").strip()
platform = spec.get("platform", "").strip()
credit = f"🎥 {creator}" + (f" on {platform}" if platform else "") if creator else ""
problems = []


def tagify(t):
    return re.sub(r"[^0-9a-z_]", "", str(t).lower().lstrip("#"))


pool = []
if a.viral and os.path.exists(a.viral):
    for row in json.load(open(a.viral)).get("hashtags", []):
        t, w = row[0], row[1]
        uses = row[2] if len(row) > 2 else 1
        if uses >= 2 and t not in ("shorts", "short", "fyp", "foryou", "foryoupage", "viral", "trending"):
            pool.append(t)                               # used by several real uploads, not generic filler



def duration(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


clips = sorted(spec.get("clips", []), key=lambda c: c.get("rank", 99))
if a.expect and len(clips) != a.expect:
    problems.append(f"batch has {len(clips)} clips, expected {a.expect}")
if [c.get("rank") for c in clips] != list(range(1, len(clips) + 1)):
    problems.append(f"ranks must be 1..{len(clips)} with no gaps: {[c.get('rank') for c in clips]}")

posts = []
for c in clips:
    tag = f"#{c.get('rank')} {c.get('file')}"
    path = os.path.join(base, c.get("file", ""))
    dur = duration(path) if os.path.exists(path) else None
    if dur is None:
        problems.append(f"{tag}: file missing or unreadable")
    elif a.max_dur and dur > a.max_dur + 0.25:
        problems.append(f"{tag}: {dur:.1f} s is over the {a.max_dur:g} s cap")
    title = (c.get("title") or "").strip()
    cap = (c.get("caption") or "").strip()
    if not title: problems.append(f"{tag}: no title")
    if not 1 <= len(cap) <= 150: problems.append(f"{tag}: caption must be 1-150 chars (has {len(cap)})")

    tags = []
    for t in [tagify(x) for x in c.get("tags", [])] + [tagify(x) for x in spec.get("base_tags", [])]:
        if t and t not in tags: tags.append(t)
    if not 4 <= len(tags) <= 8: problems.append(f"{tag}: needs 4-8 hashtags (has {len(tags)}: {tags})")
    bad = [t for t in tags if not re.fullmatch(r"[a-z0-9_]{2,30}", t)]
    if bad: problems.append(f"{tag}: bad hashtags {bad}")
    hs = " ".join("#" + t for t in tags)

    yt_title = f"{title} #shorts"
    if len(yt_title) > 100: problems.append(f"{tag}: YouTube title over 100 chars ({len(yt_title)})")
    x_post = f"{cap}\n{credit} " if credit else f"{cap} "
    x_post += " ".join("#" + t for t in tags[:2 if credit else 3])
    if len(x_post) > 280: problems.append(f"{tag}: X post over 280 chars")

    body = "\n".join(filter(None, [cap, "", credit, hs]))
    blocks = {
        "tiktok": body,
        "instagram_reels": body,
        "youtube_shorts": {"title": yt_title, "description": "\n".join(filter(None, [cap, credit, spec.get("source_url", ""), "", hs]))},
        "x": x_post,
    }
    posts.append({**c, "duration": round(dur, 1) if dur else None, "hashtags": tags, "posts": blocks})

if problems:
    print("post_kit: not post-ready:\n  " + "\n  ".join(problems), file=sys.stderr)
    sys.exit(1)

for p in posts:
    txt = os.path.splitext(os.path.join(base, p["file"]))[0] + ".txt"
    b = p["posts"]
    with open(txt, "w") as f:
        f.write(f"#{p['rank']}  {p['title']}  ({p['duration']} s, source {p.get('source_time', '?')})\n\n")
        f.write("=== TikTok ===\n" + b["tiktok"] + "\n\n")
        f.write("=== Instagram Reels ===\n" + b["instagram_reels"] + "\n\n")
        f.write("=== YouTube Shorts ===\nTitle: " + b["youtube_shorts"]["title"] + "\nDescription:\n"
                + b["youtube_shorts"]["description"] + "\n\n")
        f.write("=== X ===\n" + b["x"] + "\n")

with open(os.path.join(base, "posts.md"), "w") as f:
    f.write(f"# {len(posts)} clips: {creator}\n\n")
    if spec.get("source_url"): f.write(f"Source: {creator}, {spec['source_url']}\n\n")
    f.write("| Rank | File | Title | Source time | Length | Why |\n|---|---|---|---|---|---|\n")
    for p in posts:
        f.write(f"| {p['rank']} | {p['file']} | {p['title']} | {p.get('source_time', '')} | {p['duration']} s | "
                f"{(p.get('why') or '').replace('|', '/')} |\n")
    for p in posts:
        b = p["posts"]
        f.write(f"\n## {p['rank']}. {p['title']}\n\n`{p['file']}`\n\n**TikTok / Reels**\n\n```\n{b['tiktok']}\n```\n\n"
                f"**YouTube Shorts**: title `{b['youtube_shorts']['title']}`\n\n```\n{b['youtube_shorts']['description']}\n```\n\n"
                f"**X**\n\n```\n{b['x']}\n```\n")
    if pool:
        f.write("\n## Hashtag pool from real uploads of this creator/stream\n\n" + " ".join("#" + t for t in pool[:30]) + "\n")
json.dump({"creator": creator, "source_url": spec.get("source_url"), "posts": posts},
          open(os.path.join(base, "posts.json"), "w"), indent=1, ensure_ascii=False)
print(f"post_kit: {len(posts)} posts ready -> posts.md, posts.json and one .txt per clip", file=sys.stderr)
