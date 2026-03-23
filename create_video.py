#!/usr/bin/env python3
"""AI Latest Briefing — Explainer Video Creator
Generates a ~2.5 min video with Polly (Matthew Neural) voiceover.
"""

import boto3
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips

# ── Config ────────────────────────────────────────────────────────────────────
W, H = 1920, 1080
FPS  = 30
AWS_PROFILE  = "aws-sandbox-personal-36"
POLLY_VOICE  = "Matthew"

BASE      = Path("/Users/kobyalmog/vscode/projects/ai-agents-google-adk/ai-latest-briefing")
ASSETS    = BASE / "generated-diagrams"
OUT_DIR   = BASE / "video_output"
AUDIO_DIR = OUT_DIR / "audio"
OUT_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

# ── Colors ────────────────────────────────────────────────────────────────────
BG         = (10, 12, 35)
GRID       = (30, 35, 70)
DARK       = (15, 18, 45)
WHITE      = (255, 255, 255)
LIGHT_BLUE = (100, 180, 255)
GRAY       = (160, 170, 200)
ACCENT     = (220, 220, 255)

AGENT_COLORS = [
    (130, 80,  220),  # 1 VendorResearcher
    (40,  130, 220),  # 2 URLResolver
    (40,  180, 120),  # 3 CommunityResearcher
    (220, 120, 40),   # 4 BriefingWriter
    (220, 50,  120),  # 5 Translator
    (100, 200, 50),   # 6 Publisher
]

AGENTS = [
    ("Vendor\nResearcher",    "google_search ×5"),
    ("URL\nResolver",         "validate links"),
    ("Community\nResearcher", "HN + Reddit"),
    ("Briefing\nWriter",      "output_schema"),
    ("Translator",            "EN → Hebrew"),
    ("Publisher",             "save HTML"),
]

# ── Narration ─────────────────────────────────────────────────────────────────
SCENES = [
    ("title",
     "AI Latest Briefing. A six-step autonomous news agent, built with Google ADK."),
    ("problem",
     "Every morning, you open a handful of tabs. Vendor blogs, Hacker News, Reddit, newsletters. "
     "Trying to piece together what actually happened in AI the past few days. "
     "Half the time you miss it entirely. The other half you spend twenty minutes and still don't feel confident you have the full picture."),
    ("solution",
     "So I built an agent to do it for me. "
     "Six agents working in sequence, each with a single job. "
     "Together they research five major AI vendors, validate source links, find community reactions, "
     "and drop a clean bilingual HTML newsletter in my browser. Done."),
    ("agent1",
     "Step one. The Vendor Researcher runs five targeted Google searches, one per vendor: "
     "Anthropic, AWS, OpenAI, Google, and Azure. "
     "It collects headlines, publication dates, summaries, and every URL it finds."),
    ("agent2",
     "Step two. The URL Resolver. This step exists because Gemini grounding redirect URLs "
     "expire in thirty to sixty seconds. Running it immediately after step one — "
     "before any other processing — consistently returns ten to sixteen working article URLs."),
    ("agent3",
     "Step three. The Community Researcher searches Hacker News and Reddit "
     "for what developers are actually saying about the top stories."),
    ("agent4",
     "Step four. The Briefing Writer synthesizes everything into structured JSON "
     "using a Pydantic output schema. No regex. No parsing hacks. "
     "The data just arrives clean and ready to use."),
    ("agent5",
     "Step five. The Translator converts the entire briefing to Hebrew. "
     "Same Pydantic pattern. Product names stay in English. "
     "Dates stay as-is."),
    ("agent6",
     "Step six. The Publisher builds a self-contained HTML file "
     "with an English-Hebrew toggle and saves it locally."),
    ("state",
     "All six agents communicate through shared session state. "
     "Each agent writes one key. The next reads it. "
     "No callbacks. No message passing. State is the communication channel."),
    ("ui",
     "The ADK Web UI lets you debug every step in real time. "
     "See what each agent received, what it output, "
     "and watch session state populate live as the pipeline runs."),
    ("output",
     "The result: a newsletter with a TL;DR, color-coded vendor cards, "
     "verified source links per story, community reactions, "
     "and a full Hebrew translation — generated fresh every time you run it."),
    ("cta",
     "Full code on GitHub at kobyal slash ai-latest-briefing. "
     "Built with Google ADK and Gemini 2.5 Flash."),
]

# ── Fonts ─────────────────────────────────────────────────────────────────────
def fnt(size):
    return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)

F_HUGE  = fnt(90)
F_BIG   = fnt(68)
F_TITLE = fnt(52)
F_BODY  = fnt(38)
F_LABEL = fnt(30)
F_SMALL = fnt(24)
F_BADGE = fnt(20)

# ── Drawing helpers ────────────────────────────────────────────────────────────
def make_base():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    for x in range(0, W, 60):
        d.line([(x, 0), (x, H)], fill=GRID, width=1)
    for y in range(0, H, 60):
        d.line([(0, y), (W, y)], fill=GRID, width=1)
    return img, d

def centered_text(d, cx, y, text, fill, font):
    bbox = d.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    d.text((cx - tw // 2, y), text, fill=fill, font=font)

def draw_agent_box(d, x, y, bw, bh, color, num, name, sub, dim=False):
    alpha = 0.18 if dim else 1.0
    c  = tuple(int(color[i] * alpha + BG[i] * (1 - alpha)) for i in range(3))
    fc = tuple(int(DARK[i]  * alpha + BG[i] * (1 - alpha)) for i in range(3))
    wc = tuple(int(255 * alpha + BG[i] * (1 - alpha)) for i in range(3))
    ac = tuple(int(220 * alpha + BG[i] * (1 - alpha)) for i in range(3))
    gc = tuple(int(GRAY[i] * alpha + BG[i] * (1 - alpha)) for i in range(3))

    d.rectangle([x, y, x+bw, y+bh], fill=fc, outline=c, width=3)
    d.ellipse([x+14, y+14, x+52, y+52], fill=c)
    bx = x + 24 if num < 10 else x + 18
    d.text((bx, y+18), str(num), fill=wc, font=F_BADGE)
    for j, line in enumerate(name.split("\n")):
        d.text((x+16, y+66 + j*38), line, fill=(ac[0], ac[1], ac[2]), font=F_LABEL)
    d.text((x+16, y+bh-42), sub, fill=gc, font=F_SMALL)

def draw_pipeline(d, highlight=None, all_visible=True, visible_count=None):
    """Draw all 6 agent boxes. highlight=idx dims all others."""
    bw, bh = 240, 220
    gap    = 28
    total  = 6 * bw + 5 * gap
    sx     = (W - total) // 2
    by     = (H - bh) // 2 - 20

    n = visible_count if visible_count is not None else 6

    for i in range(n):
        x    = sx + i * (bw + gap)
        name, sub = AGENTS[i]
        color = AGENT_COLORS[i]
        dim   = (highlight is not None and i != highlight)
        draw_agent_box(d, x, by, bw, bh, color, i+1, name, sub, dim=dim)
        if i < n - 1:
            ax  = x + bw + 4
            ay  = by + bh // 2
            fa  = 0.18 if dim else 1.0
            fc  = tuple(int(color[j] * fa + BG[j] * (1-fa)) for j in range(3))
            d.polygon([(ax, ay-10), (ax+gap-5, ay), (ax, ay+10)], fill=fc)

    return by, bw, bh, sx, gap

def np_img(img):
    return np.array(img)

# ── Polly ─────────────────────────────────────────────────────────────────────
def get_audio(scene_id, text):
    path = AUDIO_DIR / f"{scene_id}.mp3"
    if not path.exists():
        session = boto3.Session(profile_name=AWS_PROFILE)
        polly   = session.client("polly", region_name="us-east-1")
        resp    = polly.synthesize_speech(
            Text=text, VoiceId=POLLY_VOICE, Engine="neural", OutputFormat="mp3"
        )
        path.write_bytes(resp["AudioStream"].read())
        print(f"  [Polly] {scene_id}")
    return str(path)

# ── Scenes ────────────────────────────────────────────────────────────────────
def scene_title(dur):
    img, d = make_base()
    logo = Image.open(ASSETS / "adk-logo.png").convert("RGBA").resize((130, 130), Image.LANCZOS)
    img.paste(logo, (W//2 - 65, H//2 - 200), logo)
    centered_text(d, W//2, H//2 - 50,  "AI Latest Briefing",          WHITE,      F_HUGE)
    centered_text(d, W//2, H//2 + 70,  "A 6-Step Multi-Agent Pipeline", LIGHT_BLUE, F_TITLE)
    centered_text(d, W//2, H//2 + 140, "Built with Google ADK + Gemini 2.5 Flash", GRAY, F_LABEL)
    return ImageClip(np_img(img)).with_duration(dur)

def scene_problem(dur):
    img, d = make_base()
    # Browser tabs
    tabs = [("anthropic.com", (100,80,200)), ("news.ycombinator.com",(255,100,0)),
            ("reddit.com/r/ml",(255,69,0)), ("x.com",(29,161,242)), ("openai.com/blog",(16,163,127))]
    for i, (label, color) in enumerate(tabs):
        tx = 160 + i * 320
        d.rectangle([tx, 140, tx+295, 185], fill=color, outline=(180,180,180), width=1)
        d.text((tx+10, 152), label, fill=WHITE, font=F_SMALL)
    # Browser body
    d.rectangle([160, 185, W-160, 660], fill=(22,26,55), outline=(70,80,130), width=2)
    # Simulated content lines
    import random; random.seed(42)
    for i in range(10):
        gy = 220 + i*42
        gw = random.randint(200, 860)
        d.rectangle([200, gy, 200+gw, gy+20], fill=(45,50,80))
    centered_text(d, W//2, 700, "20 minutes every morning.", (220,100,100), F_BODY)
    centered_text(d, W//2, 760, "And you still might miss the story.", GRAY, F_LABEL)
    return ImageClip(np_img(img)).with_duration(dur)

def scene_solution(dur):
    img, d = make_base()
    draw_pipeline(d)
    centered_text(d, W//2, H//2 + 135, "6 agents · 10 lines of orchestration code", LIGHT_BLUE, F_BODY)
    return ImageClip(np_img(img)).with_duration(dur)

AGENT_DETAILS = {
    0: ["Tool: google_search  (built-in Gemini tool)",
        "Runs 5 searches — one per vendor",
        "Output key: raw_vendor_news"],
    1: ["Tool: resolve_source_urls  (custom Python function)",
        "Runs immediately — redirect URLs expire in ~30-60s",
        "Validates + deduplicates · Output: resolved_sources"],
    2: ["Tool: google_search  (built-in)",
        "Searches HN + Reddit/MachineLearning + LocalLLaMA",
        "Output key: raw_community"],
    3: ["output_schema = BriefingContent  (Pydantic model)",
        "Forces valid JSON — no regex, no parsing",
        "Output key: briefing"],
    4: ["output_schema = HebrewBriefing  (Pydantic model)",
        "Keeps product names in English, dates as-is",
        "Output key: briefing_he"],
    5: ["Tool: build_and_save_html  (custom Python function)",
        "3 parameters only — eliminates MALFORMED_FUNCTION_CALL",
        "Saves: output/YYYY-MM-DD/briefing_HHMMSS.html"],
}

def scene_agent(idx, dur):
    img, d = make_base()
    by, bw, bh, sx, gap = draw_pipeline(d, highlight=idx)

    # Detail card below pipeline
    color  = AGENT_COLORS[idx]
    card_y = by + bh + 55
    card_h = 240
    d.rectangle([W//2 - 560, card_y, W//2 + 560, card_y + card_h],
                fill=DARK, outline=color, width=3)
    for j, line in enumerate(AGENT_DETAILS[idx]):
        d.text((W//2 - 520, card_y + 30 + j*68), f"•  {line}", fill=WHITE, font=F_LABEL)

    return ImageClip(np_img(img)).with_duration(dur)

def scene_state(dur):
    img, d = make_base()
    centered_text(d, W//2, 30, "Session State — The Communication Channel", WHITE, F_TITLE)

    keys    = ["raw_vendor_news", "resolved_sources", "raw_community", "briefing", "briefing_he"]
    writers = [0, 1, 2, 3, 4]
    readers = [[1,2,3], [3], [3], [4,5], [5]]

    kw, kh, kx = 380, 64, W//2 - 190
    row_gap = 155

    for i, (key, wi, ri) in enumerate(zip(keys, writers, readers)):
        ky = 120 + i * row_gap
        d.rectangle([kx, ky, kx+kw, ky+kh], fill=DARK, outline=LIGHT_BLUE, width=2)
        centered_text(d, kx + kw//2, ky + 16, key, LIGHT_BLUE, F_LABEL)

        # Writer arrow (left)
        wname = AGENTS[wi][0].replace("\n"," ")
        d.text((kx - 340, ky + 16), f"{wname}  →", fill=AGENT_COLORS[wi], font=F_SMALL)

        # Reader arrows (right)
        for j, r in enumerate(ri):
            rname = AGENTS[r][0].replace("\n"," ")
            ry = ky + 16 - (len(ri)-1)*16 + j*32
            d.text((kx + kw + 20, ry), f"→  {rname}", fill=AGENT_COLORS[r], font=F_SMALL)

    return ImageClip(np_img(img)).with_duration(dur)

def scene_screenshot(path, caption, dur):
    img, d = make_base()
    shot = Image.open(path).convert("RGB")
    sw, sh = shot.size
    max_w, max_h = W - 180, H - 180
    scale = min(max_w/sw, max_h/sh)
    nw, nh = int(sw*scale), int(sh*scale)
    shot = shot.resize((nw, nh), Image.LANCZOS)
    px, py = (W-nw)//2, (H-nh)//2 - 30
    img.paste(shot, (px, py))
    centered_text(d, W//2, py + nh + 20, caption, LIGHT_BLUE, F_LABEL)
    return ImageClip(np_img(img)).with_duration(dur)

def scene_cta(dur):
    img, d = make_base()
    logo = Image.open(ASSETS / "adk-logo.png").convert("RGBA").resize((130, 130), Image.LANCZOS)
    img.paste(logo, (W//2 - 65, H//2 - 230), logo)
    centered_text(d, W//2, H//2 - 70, "Built with Google ADK",               WHITE,       F_BIG)
    centered_text(d, W//2, H//2 + 30, "and Gemini 2.5 Flash",                LIGHT_BLUE,  F_TITLE)
    centered_text(d, W//2, H//2 + 130, "github.com/kobyal/ai-latest-briefing", (80,160,255), F_BODY)
    return ImageClip(np_img(img)).with_duration(dur)

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("Step 1: Generating Polly audio...")
    audio_clips = {}
    for scene_id, text in SCENES:
        path = get_audio(scene_id, text)
        audio_clips[scene_id] = AudioFileClip(path)

    print("\nStep 2: Building video scenes...")
    clips = []

    for scene_id, text in SCENES:
        audio  = audio_clips[scene_id]
        dur    = audio.duration + 0.4   # small pause after each segment
        print(f"  {scene_id:<20} {dur:.1f}s")

        if   scene_id == "title":    clip = scene_title(dur)
        elif scene_id == "problem":  clip = scene_problem(dur)
        elif scene_id == "solution": clip = scene_solution(dur)
        elif scene_id == "agent1":   clip = scene_agent(0, dur)
        elif scene_id == "agent2":   clip = scene_agent(1, dur)
        elif scene_id == "agent3":   clip = scene_agent(2, dur)
        elif scene_id == "agent4":   clip = scene_agent(3, dur)
        elif scene_id == "agent5":   clip = scene_agent(4, dur)
        elif scene_id == "agent6":   clip = scene_agent(5, dur)
        elif scene_id == "state":    clip = scene_state(dur)
        elif scene_id == "ui":       clip = scene_screenshot(
            ASSETS / "adk_ui_screenshot.png", "ADK Web UI — real-time debugging", dur)
        elif scene_id == "output":   clip = scene_screenshot(
            ASSETS / "output_screenshot.png", "Final output — bilingual HTML newsletter", dur)
        elif scene_id == "cta":      clip = scene_cta(dur)

        clip = clip.with_audio(audio)
        clips.append(clip)

    print("\nStep 3: Assembling final video...")
    final    = concatenate_videoclips(clips, method="compose")
    out_path = str(OUT_DIR / "ai_latest_briefing_explainer.mp4")
    final.write_videofile(out_path, fps=FPS, codec="libx264", audio_codec="aac", logger="bar")

    total = sum(audio_clips[s].duration for s, _ in SCENES)
    print(f"\nDone! Total: {total:.0f}s  →  {out_path}")

if __name__ == "__main__":
    main()
