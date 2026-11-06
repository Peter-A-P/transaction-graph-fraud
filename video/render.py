"""Render the site's short film: video/ring.html drawn at 30 frames a second for 13.5 seconds,
each frame screenshot by a headless browser and the frames encoded to H.264 MP4.

    uv run --no-project --with playwright --with imageio-ffmpeg --with numpy python video/render.py

The page reads site/data.json (written by `consort site`), so run that first when the results
change. It uses the machine's own Edge (Playwright's `channel`; pass another, such as
`chromium` in CI, as the first argument) and imageio-ffmpeg's bundled ffmpeg, so none needs
installing. Writes video/out/ring.mp4 (silent), video/out/ring-sound.mp4 (with video/sound.py's
track) and a poster frame, and copies them to site/video/ for the page. Both folders are
gitignored: the website workflow renders the film on every deploy.
poster frame beside them.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FPS = 30
# The story is drawn on a 9.2-second schedule in ring.html; each scene plays at its own speed
# (story start, story end, how many times slower): one transfer, the ring drawing in around it,
# both models scoring it. The result screen after it plays at its own pace.
SEGMENTS = [(0.0, 3.0, 1.3), (3.0, 6.2, 1.0), (6.2, 9.2, 1.3)]
SPLIT = 9.2
STORY = sum((b - a) * k for a, b, k in SEGMENTS)
# Then the camera pulls back from the ring to the network around it, for ZOOM seconds.
ZOOM = 4.2
SECONDS = STORY + ZOOM + (13.5 - SPLIT)
POSTER_AT = STORY - 0.3 * SEGMENTS[-1][2]  # the ring with every transfer scored


def main() -> None:
    import imageio_ffmpeg  # type: ignore[import-untyped]
    from playwright.sync_api import sync_playwright

    data = json.loads((ROOT / "site" / "data.json").read_text(encoding="utf8"))
    out = ROOT / "video" / "out"
    out.mkdir(parents=True, exist_ok=True)
    frames = Path(tempfile.mkdtemp(prefix="ring-frames-"))
    # --site: the cut for the page itself, without the address on the last screen (ring-site*).
    site = "--site" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    channel = args[0] if args else "msedge"
    stem = "ring-site" if site else "ring"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel=channel)
            page = browser.new_page(viewport={"width": 1080, "height": 1080})
            page.goto((ROOT / "video" / "ring.html").as_uri())
            page.evaluate("d => { window.D = d; }", data)
            page.evaluate(
                "k => { window.TIMING = k; }", {"segments": SEGMENTS, "split": SPLIT, "zoom": ZOOM, "site": site}
            )
            page.evaluate("document.fonts.ready")
            n = round(SECONDS * FPS)
            for i in range(n):
                page.evaluate("t => render(t)", i / FPS)
                page.screenshot(path=str(frames / f"f{i:04d}.png"))
            page.evaluate("t => render(t)", POSTER_AT)
            page.screenshot(path=str(out / f"{stem}-poster.jpg"), type="jpeg", quality=88)
            browser.close()
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-y",
                "-framerate",
                str(FPS),
                "-i",
                str(frames / "f%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "20",
                "-preset",
                "slow",
                "-movflags",
                "+faststart",
                str(out / f"{stem}.mp4"),
            ],
            check=True,
        )
        # The sound version: the same frames, with the track timed to them.
        sys.path.insert(0, str(Path(__file__).parent))
        import sound

        left, right = sound.track(data["ring"], SECONDS, SEGMENTS, SPLIT, ZOOM)
        sound.write_wav(frames / "track.wav", left, right)
        subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-y",
                "-i",
                str(out / f"{stem}.mp4"),
                "-i",
                str(frames / "track.wav"),
                "-c:v",
                "copy",
                "-af",
                "loudnorm=I=-16:TP=-1.5:LRA=11",
                "-ar",
                "48000",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                "-movflags",
                "+faststart",
                str(out / f"{stem}-sound.mp4"),
            ],
            check=True,
        )
    finally:
        shutil.rmtree(frames, ignore_errors=True)
    page_dir = ROOT / "site" / "video"
    page_dir.mkdir(parents=True, exist_ok=True)
    names = [f"{stem}.mp4", f"{stem}-sound.mp4", f"{stem}-poster.jpg"]
    for name in names:
        shutil.copy2(out / name, page_dir / name)
    print(f"wrote {', '.join(names)} to {out}, copied to {page_dir}")


if __name__ == "__main__":
    main()
