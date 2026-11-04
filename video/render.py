"""Render the site's short film: video/ring.html drawn at 30 frames a second for 13.5 seconds,
each frame screenshot by a headless browser and the frames encoded to H.264 MP4.

    uv run --no-project --with playwright --with imageio-ffmpeg --with numpy python video/render.py

The page reads site/data.json (written by `consort site`), so run that first when the results
change. It uses the machine's own Edge or Chrome (Playwright's `channel`), so no browser is
downloaded, and imageio-ffmpeg's bundled ffmpeg, so none needs installing. Writes
site/video/ring.mp4 (silent), site/video/ring-sound.mp4 (with video/sound.py's track) and a
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
# The story (one transfer, the ring, both models scoring it) is drawn on a 9.2-second schedule
# in ring.html and played STRETCH times slower; the result screen after it plays at its own pace.
STRETCH = 1.3
SPLIT = 9.2
# Then the camera pulls back from the ring to the network around it, for ZOOM seconds.
ZOOM = 4.2
SECONDS = 13.5 + SPLIT * (STRETCH - 1) + ZOOM
POSTER_AT = 8.9 * STRETCH  # the ring with every transfer scored


def main() -> None:
    import imageio_ffmpeg  # type: ignore[import-untyped]
    from playwright.sync_api import sync_playwright

    data = json.loads((ROOT / "site" / "data.json").read_text(encoding="utf8"))
    out = ROOT / "site" / "video"
    out.mkdir(parents=True, exist_ok=True)
    frames = Path(tempfile.mkdtemp(prefix="ring-frames-"))
    channel = sys.argv[1] if len(sys.argv) > 1 else "msedge"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel=channel)
            page = browser.new_page(viewport={"width": 1080, "height": 1080})
            page.goto((ROOT / "video" / "ring.html").as_uri())
            page.evaluate("d => { window.D = d; }", data)
            page.evaluate(
                "k => { window.TIMING = k; }", {"stretch": STRETCH, "split": SPLIT, "zoom": ZOOM}
            )
            page.evaluate("document.fonts.ready")
            n = round(SECONDS * FPS)
            for i in range(n):
                page.evaluate("t => render(t)", i / FPS)
                page.screenshot(path=str(frames / f"f{i:04d}.png"))
            page.evaluate("t => render(t)", POSTER_AT)
            page.screenshot(path=str(out / "ring-poster.jpg"), type="jpeg", quality=88)
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
                str(out / "ring.mp4"),
            ],
            check=True,
        )
        # The sound version: the same frames, with the track timed to them.
        sys.path.insert(0, str(Path(__file__).parent))
        import sound

        left, right = sound.track(data["ring"], SECONDS, STRETCH, SPLIT, ZOOM)
        sound.write_wav(frames / "track.wav", left, right)
        subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-y",
                "-i",
                str(out / "ring.mp4"),
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
                str(out / "ring-sound.mp4"),
            ],
            check=True,
        )
    finally:
        shutil.rmtree(frames, ignore_errors=True)
    print(f"wrote {out / 'ring.mp4'}, {out / 'ring-sound.mp4'} and {out / 'ring-poster.jpg'}")


if __name__ == "__main__":
    main()
