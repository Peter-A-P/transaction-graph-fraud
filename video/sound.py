"""The film's sound track, synthesised: a soft pad under the whole film, and a sound for each
thing that happens on screen, timed by the same schedule video/ring.html draws to.

- the two accounts appear: two soft notes; the transfer draws: a rising sweep;
- the row model's score: a low tone, and a muted thud for "not flagged";
- the ring draws in: a quiet note as each account appears;
- the row model's pass: a dull tick for each transfer it flags;
- graph features' pass: a rising scale, one bright note for each transfer they flag;
- the result: a swell into a brighter chord, a note for each figure card and a final chord.

48 kHz stereo, peak at -1 dBFS, a mean near -17 dB, like project 09's film. numpy only.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

SR = 48_000
Arr = npt.NDArray[np.float64]

# A minor pentatonic from A4 up: the graph features' pass climbs it.
SCALE = [440.0, 523.25, 587.33, 659.26, 783.99, 880.0, 1046.5, 1174.66, 1318.51, 1567.98, 1760.0]


def _t(n: int) -> Arr:
    return np.arange(n) / SR


def pluck(freq: float, dur: float = 0.9, decay: float = 5.0, bright: float = 0.35) -> Arr:
    """A plucked note: a fundamental with two soft harmonics and an exponential decay."""
    t = _t(int(dur * SR))
    tone = (
        np.sin(2 * np.pi * freq * t)
        + bright * np.sin(4 * np.pi * freq * t)
        + 0.12 * np.sin(6 * np.pi * freq * t)
    )
    env = np.minimum(t / 0.006, 1.0) * np.exp(-decay * t)
    out: Arr = tone * env
    return out


def thud(freq: float = 82.0, dur: float = 0.45) -> Arr:
    t = _t(int(dur * SR))
    out: Arr = (
        np.sin(2 * np.pi * freq * t * (1 - 0.25 * t)) * np.minimum(t / 0.004, 1) * np.exp(-9 * t)
    )
    return out


def tick(rng: np.random.Generator, dur: float = 0.05) -> Arr:
    n = int(dur * SR)
    noise = rng.standard_normal(n)
    smooth = np.convolve(noise, np.ones(12) / 12, mode="same")  # a soft click, not a hiss
    out: Arr = smooth * np.exp(-60 * _t(n))
    return out


def sweep(f0: float, f1: float, dur: float) -> Arr:
    t = _t(int(dur * SR))
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * dur))
    env = np.sin(np.pi * t / dur) ** 2
    out: Arr = np.sin(phase) * env
    return out


def pad(dur: float, chord: list[float], fade: float) -> Arr:
    t = _t(int(dur * SR))
    out = np.zeros_like(t)
    for i, f in enumerate(chord):
        lfo = 0.75 + 0.25 * np.sin(2 * np.pi * (0.13 + 0.05 * i) * t + i)
        out += lfo * (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * f * 1.003 * t))
    env = np.minimum(np.minimum(t / fade, 1.0), np.minimum((dur - t) / fade, 1.0))
    res: Arr = out / len(chord) * env
    return res


def track(ring: dict[str, Any], seconds: float) -> tuple[Arr, Arr]:
    """Left and right channels for the film of `ring` (site/data.json's ring)."""
    n = int(seconds * SR)
    left, right = np.zeros(n), np.zeros(n)
    rng = np.random.default_rng(17)

    def add(sig: Arr, at: float, gain: float, pan: float = 0.0) -> None:
        i = int(at * SR)
        j = min(n, i + len(sig))
        if i >= n:
            return
        left[i:j] += sig[: j - i] * gain * (1 - pan) ** 0.5
        right[i:j] += sig[: j - i] * gain * (1 + pan) ** 0.5

    # The pad: A minor while the ring is told, then a brighter D major under the result.
    add(pad(10.2, [110.0, 164.81, 220.0, 261.63], 1.2), 0.0, 0.10)
    add(pad(seconds - 9.4, [146.83, 220.0, 293.66, 369.99], 0.9), 9.4, 0.10)

    # Scene 1.
    add(pluck(329.63, decay=4), 0.25, 0.22, -0.4)
    add(pluck(392.0, decay=4), 0.40, 0.22, 0.4)
    add(sweep(260, 620, 0.95), 0.5, 0.10)
    add(pluck(196.0, 0.6, 7, 0.1), 1.55, 0.18)
    add(thud(), 2.35, 0.40)

    # Scene 2: an account appears at 3.4 + (its number mod 7) * 0.12 s, as ring.html has it.
    names = sorted({a["name"] for a in ring["accounts"]} - {"A3", "A15"}, key=lambda s: int(s[1:]))
    for k, name in enumerate(names):
        at = 3.4 + (int(name[1:]) % 7) * 0.12 + 0.15
        add(pluck(SCALE[k % 5] / 2, 0.5, 8, 0.2), at, 0.07, -0.6 + 1.2 * k / max(len(names) - 1, 1))
    add(sweep(180, 360, 2.0), 3.7, 0.05)

    # Scene 3: an arrow changes colour when its pass reaches its place in the scored list.
    scored = [x for x in ring["transfers"] if x["scored"]]
    for i, x in enumerate(scored):
        if x["arms"]["R"]["flagged"]:
            add(tick(rng), 6.4 + 1.2 * i / len(scored), 0.55, -0.3)
    step = 0
    for i, x in enumerate(scored):
        if x["arms"]["G"]["flagged"]:
            add(
                pluck(SCALE[min(step, len(SCALE) - 1)], 1.0, 4.5, 0.4),
                7.7 + 1.2 * i / len(scored),
                0.13,
                0.3,
            )
            step += 1

    # Scene 4.
    add(sweep(220, 880, 0.9), 9.3, 0.06)
    for k, at in enumerate((10.45, 10.8)):
        add(pluck([587.33, 739.99][k], 0.7, 6), at, 0.14, [-0.3, 0.3][k])
    for f in (293.66, 369.99, 440.0, 587.33, 739.99):
        add(pluck(f, 2.4, 1.6, 0.25), 11.35, 0.09)

    # A little room: three soft echoes.
    for d, g in ((0.083, 0.22), (0.131, 0.15), (0.197, 0.1)):
        k = int(d * SR)
        left[k:] += g * right[:-k]
        right[k:] += g * left[:-k]
    tail = np.minimum(1.0, (seconds - _t(n)) / 0.6)
    left, right = left * tail, right * tail
    peak = max(np.abs(left).max(), np.abs(right).max())
    scale = 10 ** (-1 / 20) / peak
    return left * scale, right * scale


def write_wav(path: Path, left: Arr, right: Arr) -> None:
    pcm = (np.stack([left, right], axis=1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
