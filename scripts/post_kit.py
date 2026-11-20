#!/usr/bin/env python3
"""Make every rendered clip post-ready: caption, hashtags, title and source
credit per platform, checked against each platform's limits.

  post_kit.py clips.json [--viral viral.json] [--sort] [--expect 10] [--max-dur 30]

clips.json (you write it after the newcomer review; paths are relative to it):
  {
    "creator": "StreamerName",               # credited in every post
    "source_url": "https://www.twitch.tv/videos/123456789",
    "platform": "Twitch",                    # where the source lives (credit line)
    "base_tags": ["gamename", "streamername"],  # evergreen tags (see the format guide)
    "clips": [
      {"rank": 1, "file": "queue/01_show_up_in_style.mp4",
       "title": "He said he drives better on his phone",   # short, spoiler-free
       "caption": "chat said he drives better while texting so he proved it 💀",
       "tags": ["texting", "baddriver", "gamenamefunny"],  # moment-specific, relatable
       "source_time": "3:37:01", "source_start": 13021, "source_end": 13059,   # seconds in the source
       "kind": "game", "why": "..."},
      ...
    ]
  }

Hashtags per post = the clip's own tags, then base_tags, de-duplicated,
lower-cased, 4-8 in total. Pick the clip's own tags yourself, per moment:
relevance can't be automated (a creator's top tag may be another game).
--viral lists viral.json's pool (tags that several real uploads of this
creator/stream used, ranked by views) at the end of posts.md to choose from.

Proven vs fresh (needs --viral and each clip's source_start/source_end):
  proven  the clip overlaps a moment other people already clipped/posted that
          blew up: views in the top 5% of this source's posted moments, at least
          --min-proven-views (50) and at most --proven-views (1000) needed, so
          "blew up" scales with the creator's size.
          The evidence (views, titles) is written into the post files.
  fresh   nobody has posted this moment yet (or not enough to call it proven).
With --sort the files move into queue/proven/ and queue/fresh/ (names and
overall ranks unchanged); posts.md lists both groups separately.

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
ap.add_argument("--proven-views", type=int, default=1000,
                help="views at which a posted moment counts as blown up regardless of the creator's size")
ap.add_argument("--min-proven-views", type=int, default=50, help="never call a moment proven below this many views")
ap.add_argument("--sort", action="store_true", help="move clips into proven/ and fresh/ subfolders")
a = ap.parse_args()

base = os.path.dirname(os.path.abspath(a.clips))
spec = json.load(open(a.clips))
creator = spec.get("creator", "").strip()
platform = spec.get("platform", "").strip()
credit = f"🎥 {creator}" + (f" on {platform}" if platform else "") if creator else ""
problems = []


def tagify(t):
    return re.sub(r"[^0-9a-z_]", "", str(t).lower().lstrip("#"))


# generic reach-bait tags: no topical signal, often filtered by platforms
FILLER = {"shorts", "short", "youtubeshorts", "ytshorts", "fyp", "foryou", "foryoupage", "foryourpage", "viral",
          "viralvideo", "viralmoment", "viralshorts", "trending", "trend", "explore", "explorepage", "reels",
          "tiktok", "subscribe", "like", "epic", "insane", "funny", "memes"}
pool, moments = [], []
viral = json.load(open(a.viral)) if a.viral and os.path.exists(a.viral) else None
if viral:
    moments = viral.get("moments", [])
    for row in viral.get("hashtags", []):
        t, w = row[0], row[1]
        uses = row[2] if len(row) > 2 else 1
        if uses >= 2 and t not in FILLER:
            pool.append(t)                               # used by several real uploads, not generic filler



def duration(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def proven_threshold():
    if not moments:
        return None
    v = sorted(m.get("views") or 0 for m in moments)
    top5 = v[int(0.95 * (len(v) - 1))]
    # top 5% of this source's posted moments, at least --min-proven-views, at most --proven-views:
    # scales with the creator (most viewer clips get a handful of views)
    return min(a.proven_views, max(top5, a.min_proven_views))


THRESH = proven_threshold()


def classify(c):
    """('proven', evidence) when the clip overlaps a moment others posted that blew up."""
    if viral is None:
        return "fresh", "no viral scan for this batch"
    s0, s1 = c.get("source_start"), c.get("source_end")
    if s0 is None or s1 is None:
        problems.append(f"#{c.get('rank')} {c.get('file')}: needs source_start/source_end (seconds) to sort proven/fresh")
        return "fresh", ""
    hits = [m for m in moments if m["t0"] < s1 and m["t1"] > s0]
    best = max(hits, key=lambda m: m.get("views") or 0, default=None)
    if best and (best.get("views") or 0) >= THRESH:
        titles = "; ".join(f"\"{x['title'][:50]}\" ({x.get('views') or 0:,} views, {x['source'].replace('_', ' ')})"
                           for x in best["sources"][:3])
        return "proven", f"already posted {best['uploads']}x, {best['views']:,} views: {titles}"
    if best:
        return "fresh", f"clipped by others but small ({best['views']:,} views, below {THRESH:,})"
    return "fresh", "nobody has posted this moment yet"


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
    group, evidence = classify(c)
    posts.append({**c, "duration": round(dur, 1) if dur else None, "hashtags": tags, "posts": blocks,
                  "group": group, "evidence": evidence})

if problems:
    print("post_kit: not post-ready:\n  " + "\n  ".join(problems), file=sys.stderr)
    sys.exit(1)

if a.sort:
    for p in posts:
        src = os.path.join(base, p["file"])
        d = os.path.join(os.path.dirname(src), p["group"])
        if os.path.basename(os.path.dirname(src)) in ("proven", "fresh"):
            d = os.path.join(os.path.dirname(os.path.dirname(src)), p["group"])
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, os.path.basename(src))
        if os.path.abspath(src) != os.path.abspath(dst):
            old_txt = os.path.splitext(src)[0] + ".txt"
            if os.path.exists(old_txt): os.remove(old_txt)
            os.replace(src, dst)
        p["file"] = os.path.relpath(dst, base)

for p in posts:
    txt = os.path.splitext(os.path.join(base, p["file"]))[0] + ".txt"
    b = p["posts"]
    with open(txt, "w") as f:
        f.write(f"#{p['rank']}  {p['title']}  ({p['duration']} s, source {p.get('source_time', '?')})\n")
        f.write(f"{p['group'].upper()}: {p['evidence']}\n\n")
        f.write("=== TikTok ===\n" + b["tiktok"] + "\n\n")
        f.write("=== Instagram Reels ===\n" + b["instagram_reels"] + "\n\n")
        f.write("=== YouTube Shorts ===\nTitle: " + b["youtube_shorts"]["title"] + "\nDescription:\n"
                + b["youtube_shorts"]["description"] + "\n\n")
        f.write("=== X ===\n" + b["x"] + "\n")

with open(os.path.join(base, "posts.md"), "w") as f:
    f.write(f"# {len(posts)} clips: {creator}\n\n")
    if spec.get("source_url"): f.write(f"Source: {creator}, {spec['source_url']}\n\n")
    groups = [("proven", "Proven: similar to posts that already blew up"),
              ("fresh", "Fresh: nobody has posted these yet")]
    if THRESH is not None:
        f.write(f"Proven = overlaps a moment others posted with at least {THRESH:,} views.\n\n")
    for g, head in groups:
        rows = [p for p in posts if p["group"] == g]
        f.write(f"## {head} ({len(rows)})\n\n")
        if not rows:
            f.write("None in this batch.\n\n"); continue
        f.write("| Rank | File | Title | Source time | Length | Why | Evidence |\n|---|---|---|---|---|---|---|\n")
        for p in rows:
            f.write(f"| {p['rank']} | {p['file']} | {p['title']} | {p.get('source_time', '')} | {p['duration']} s | "
                    f"{(p.get('why') or '').replace('|', '/')} | {p['evidence'].replace('|', '/')} |\n")
        f.write("\n")
    for p in posts:
        b = p["posts"]
        f.write(f"\n### {p['rank']}. {p['title']} ({p['group']})\n\n`{p['file']}`\n\n**TikTok / Reels**\n\n```\n{b['tiktok']}\n```\n\n"
                f"**YouTube Shorts**: title `{b['youtube_shorts']['title']}`\n\n```\n{b['youtube_shorts']['description']}\n```\n\n"
                f"**X**\n\n```\n{b['x']}\n```\n")
    if pool:
        f.write("\n### Hashtag pool from real uploads of this creator/stream\n\n" + " ".join("#" + t for t in pool[:30]) + "\n")
json.dump({"creator": creator, "source_url": spec.get("source_url"), "posts": posts},
          open(os.path.join(base, "posts.json"), "w"), indent=1, ensure_ascii=False)
n_p = sum(p["group"] == "proven" for p in posts)
print(f"post_kit: {len(posts)} posts ready ({n_p} proven, {len(posts) - n_p} fresh) -> posts.md, posts.json "
      "and one .txt per clip", file=sys.stderr)
