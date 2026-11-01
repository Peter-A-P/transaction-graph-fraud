"""Plain punctuation in everything written here: no typographic dashes or curly quotes."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# En and em dashes, the other Unicode dashes, and curly single and double quotes.
BANNED = {
    chr(c) for c in (0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2018, 0x2019, 0x201C, 0x201D)
}
SUFFIXES = {".py", ".md", ".toml", ".yml", ".yaml", ".json", ".txt", ".svg", ".html"}


def _tracked() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
    return [ROOT / p for p in out.stdout.splitlines() if Path(p).suffix in SUFFIXES]


def test_no_typographic_punctuation() -> None:
    bad = []
    for path in _tracked():
        if not path.exists():
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            hit = BANNED & set(line)
            if hit:
                bad.append(f"{path.relative_to(ROOT)}:{n}: {sorted(hex(ord(c)) for c in hit)}")
    assert not bad, "\n".join(bad)
