"""Capture the README screenshots and demo GIF from the running app (reproducible).

Launches ``app.py`` on a free port (real model, CPU), drives it with a headless Chromium
(Playwright) and writes:

* ``docs/screenshots/*.png`` (1440x900 viewport, light theme, optimised),
* ``docs/demo.gif`` (built from a Playwright screen recording with ffmpeg palettegen/paletteuse).

Usage::

    uv pip install --python .venv/bin/python playwright
    .venv/bin/python -m playwright install chromium
    .venv/bin/python scripts/capture_media.py              # everything
    .venv/bin/python scripts/capture_media.py --only shots # or: --only gif

Requires ffmpeg on PATH for the GIF.
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "docs" / "screenshots"
GIF = ROOT / "docs" / "demo.gif"
EX = ROOT / "examples"
W, H = 1440, 900

# Example targets by file name (gallery positions are looked up in examples/labels.csv).
AI_EXAMPLE = "ai_veo_3_volcano_flowerpot.jpg"  # Veo 3, unseen generator
REAL_EXAMPLE = "real_pexels_shark.jpg"
UNCERTAIN_EXAMPLE = "hard_real_docci_hyacinths.jpg"  # real photo inside the 0.35-0.65 band
HARD_EXAMPLE = "hard_ai_midjourney_6_canoe_lagoon.jpg"  # Midjourney 6, confident miss
SMALL_SOURCE = "real_pexels_shark.jpg"  # downscaled to 160 px for the "too small" case


def gallery_slot(name: str) -> tuple[str, int]:
    """(gallery elem_id, index) of an example, mirroring the app: regular examples, then hard cases."""
    import csv

    rows = list(csv.DictReader(open(EX / "labels.csv", encoding="utf-8")))
    main = [r["file"] for r in rows if not (r.get("hard_case") or "").strip()]
    hard = [r["file"] for r in rows if (r.get("hard_case") or "").strip()]
    if name in main:
        return "tl-examples", main.index(name)
    if name in hard:
        return "tl-hard", hard.index(name)
    raise SystemExit(f"{name} is not in examples/labels.csv")


def batch_files() -> list[Path]:
    """Eight regular examples plus the hard cases, for the batch screenshot."""
    import csv

    rows = list(csv.DictReader(open(EX / "labels.csv", encoding="utf-8")))
    main = [r for r in rows if not (r.get("hard_case") or "").strip()][:8]
    hard = [r for r in rows if (r.get("hard_case") or "").strip()]
    return [EX / r["file"] for r in main + hard]


# ----------------------------------------------------------------------------- app process
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_app(port: int, log_path: Path) -> subprocess.Popen:
    env = {**os.environ, "PORT": str(port), "GRADIO_ANALYTICS_ENABLED": "False"}
    env.pop("TRUTHLENS_FORCE_STUB", None)
    log = open(log_path, "w")
    proc = subprocess.Popen([sys.executable, str(ROOT / "app.py")], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}/"
    for _ in range(600):  # model load can take a while on first run
        if proc.poll() is not None:
            sys.exit(f"app exited early; see {log_path}")
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return proc
        except Exception:
            time.sleep(1)
    proc.terminate()
    sys.exit("app did not become ready")


# ----------------------------------------------------------------------------- page helpers
def open_app(page, port: int) -> None:
    page.goto(f"http://127.0.0.1:{port}/?__theme=light", wait_until="domcontentloaded", timeout=120_000)
    page.wait_for_selector("#tl-examples img", timeout=60_000)
    page.wait_for_timeout(800)


def card_html(page) -> str:
    return page.locator("#tl-result").inner_html()


def wait_result(page, before: str, evidence: bool = True) -> None:
    page.wait_for_function(
        "(b) => { const el = document.querySelector('#tl-result');"
        " return el && el.innerHTML !== b && el.querySelector('.tl-badge'); }",
        arg=before, timeout=120_000)
    if evidence:
        page.wait_for_function(
            "() => [...document.querySelectorAll('img')].filter(i => i.closest('[class*=image]') && i.naturalWidth > 0"
            " && i.getBoundingClientRect().width > 150).length >= 3", timeout=60_000)
    page.wait_for_timeout(900)


def pick(page, gallery: str, index: int, evidence: bool = True) -> None:
    before = card_html(page)
    page.locator(f"#{gallery} .thumbnail-item, #{gallery} button:has(img)").nth(index).click()
    wait_result(page, before, evidence)


def scroll_to(page, selector: str, offset: int = 16) -> None:
    """Scroll so the element's top sits ``offset`` px below the top of the viewport."""
    y = page.evaluate("(s) => document.querySelector(s).getBoundingClientRect().top + window.scrollY", selector)
    page.evaluate("(y) => window.scrollTo(0, y)", max(0, int(y) - offset))
    page.wait_for_timeout(400)


def upload(page, path: Path, evidence: bool = True) -> None:
    before = card_html(page)
    page.locator("#tl-input input[type=file]").set_input_files(str(path))
    wait_result(page, before, evidence)


def tab(page, name: str) -> None:
    page.get_by_role("tab", name=name).click()
    page.wait_for_timeout(700)


def shot(page, name: str, clip: dict | None = None, full: bool = False) -> Path:
    out = SHOTS / name
    page.screenshot(path=str(out), clip=clip, full_page=full or clip is not None)
    optimise(out)
    print(f"  {out.relative_to(ROOT)}  {out.stat().st_size / 1e3:.0f} kB")
    return out


def optimise(path: Path, max_w: int = 1440) -> None:
    im = Image.open(path).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    im.save(path, optimize=True)
    if path.stat().st_size > 900_000:  # quantise large captures (UI colours survive 256-colour palettes)
        im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(path, optimize=True)


def small_image(tmp: Path) -> Path:
    p = tmp / "small_upload.jpg"
    Image.open(EX / SMALL_SOURCE).resize((160, 160), Image.LANCZOS).save(p, quality=92)
    return p


# ----------------------------------------------------------------------------- captures
def capture_shots(pw, port: int, tmp: Path) -> None:
    SHOTS.mkdir(parents=True, exist_ok=True)
    browser = pw.chromium.launch()
    ctx = browser.new_context(viewport={"width": W, "height": H}, color_scheme="light", device_scale_factor=1)
    page = ctx.new_page()
    open_app(page, port)

    pick(page, *gallery_slot(AI_EXAMPLE))
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(400)
    shot(page, "detect_ai.png")
    scroll_to(page, ".tl-why", 260)  # the reasons, heatmap and spectrum for the same image
    shot(page, "evidence.png")
    for example, name in ((REAL_EXAMPLE, "detect_real.png"), (UNCERTAIN_EXAMPLE, "uncertain.png"),
                          (HARD_EXAMPLE, "hard_case.png")):
        pick(page, *gallery_slot(example))
        scroll_to(page, "#tl-result")
        shot(page, name)
    upload(page, small_image(tmp))
    scroll_to(page, "#tl-result")
    shot(page, "too_small.png")

    tab(page, "Batch")
    files = batch_files()
    page.locator("input[type=file][multiple]").first.set_input_files([str(f) for f in files])
    page.wait_for_timeout(1500)
    page.get_by_role("button", name="Analyze all").click()
    page.wait_for_selector(".tl-batch-summary", timeout=180_000)
    page.wait_for_timeout(1200)
    shot(page, "batch.png")

    tab(page, "How it works")
    page.wait_for_selector(".tl-takeaways", timeout=30_000)
    page.wait_for_timeout(800)
    top = page.locator(".tl-takeaways").bounding_box()
    rob = page.locator(".tl-table", has_text="JPEG").last.bounding_box()
    y0 = max(0, top["y"] + page.evaluate("window.scrollY") - 24)
    y1 = rob["y"] + page.evaluate("window.scrollY") + rob["height"] + 60
    shot(page, "how_it_works.png", clip={"x": 0, "y": y0, "width": W, "height": y1 - y0})

    tab(page, "Limitations & responsible use")
    page.evaluate("window.scrollTo(0, 0)")
    shot(page, "limitations.png")
    ctx.close()
    browser.close()


def smooth_scroll_to(page, selector: str, offset: int = 16, ms: int = 1000) -> None:
    y = page.evaluate("(s) => document.querySelector(s).getBoundingClientRect().top + window.scrollY", selector)
    page.evaluate("(y) => window.scrollTo({top: y, behavior: 'smooth'})", max(0, int(y) - offset))
    page.wait_for_timeout(ms)


def capture_gif(pw, port: int, tmp: Path) -> None:
    vid_dir = tmp / "video"
    browser = pw.chromium.launch()
    ctx = browser.new_context(viewport={"width": W, "height": H}, color_scheme="light",
                              record_video_dir=str(vid_dir), record_video_size={"width": W, "height": H})
    page = ctx.new_page()
    t_start = time.monotonic()
    open_app(page, port)
    t_ready = time.monotonic() - t_start
    page.wait_for_timeout(1000)  # landing
    page.mouse.move(300, 600)
    pick(page, *gallery_slot(AI_EXAMPLE))  # unseen-generator example
    page.wait_for_timeout(1800)  # verdict, gauge and reasons
    smooth_scroll_to(page, ".tl-why", 260)  # heatmap + spectrum
    page.wait_for_timeout(1400)
    pick(page, *gallery_slot(HARD_EXAMPLE))  # hard case: a confident miss
    smooth_scroll_to(page, "#tl-result", 16, 700)
    page.wait_for_timeout(2000)
    tab(page, "How it works")
    page.wait_for_selector(".tl-takeaways", timeout=30_000)
    smooth_scroll_to(page, ".tl-takeaways", 20, 900)
    page.wait_for_timeout(2200)
    video = page.video.path()
    ctx.close()
    browser.close()

    palette = tmp / "palette.png"
    trim = f"{max(0.0, t_ready - 0.2):.2f}"
    vf = "fps=10,scale=1040:-1:flags=lanczos"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", trim, "-i", str(video), "-vf",
                    f"{vf},palettegen=max_colors=192:stats_mode=diff", str(palette)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", trim, "-i", str(video), "-i", str(palette),
                    "-lavfi", f"{vf}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
                    "-loop", "0", str(GIF)], check=True)
    print(f"  {GIF.relative_to(ROOT)}  {GIF.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["shots", "gif"], default=None)
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright

    if args.only != "shots" and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH")
    tmp = Path(tempfile.mkdtemp(prefix="truthlens_media_"))
    port = free_port()
    print(f"starting app on port {port} (log: {tmp / 'app.log'})")
    proc = start_app(port, tmp / "app.log")
    try:
        with sync_playwright() as pw:
            if args.only in (None, "shots"):
                capture_shots(pw, port, tmp)
            if args.only in (None, "gif"):
                capture_gif(pw, port, tmp)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
