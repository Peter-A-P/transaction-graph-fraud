"""The demo site as committed: its host configuration is the one `consort.report.host` writes,
its pages use nothing that configuration's content security policy refuses, its script finds
every element it draws into, and its diagrams are the documentation's. A page that breaks any
of these fails here rather than as a blank chart in a visitor's browser."""

from __future__ import annotations

import json
import re
from pathlib import Path

from consort.report import host

ROOT = Path(__file__).parents[1]
SITE = ROOT / "site"
PAGES = [SITE / "index.html", SITE / "architecture.html"]


def test_the_committed_configuration_is_the_one_the_code_writes() -> None:
    committed = json.loads((SITE / "staticwebapp.config.json").read_text(encoding="utf8"))
    assert committed == host.config()
    csp = committed["globalHeaders"]["Content-Security-Policy"]
    assert "default-src 'none'" in csp
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp


def test_the_pages_use_nothing_the_policy_refuses() -> None:
    for page in PAGES:
        text = page.read_text(encoding="utf8")
        assert not re.search(r"<script>|<script(?![^>]*\bsrc=)[^>]*>", text), (
            page.name
        )  # inline script
        assert "<style" not in text, page.name
        assert not re.search(r"\sstyle=", text), page.name
        assert not re.search(r"\son[a-z]+=", text), page.name  # inline event handlers
        for ref in re.findall(r'(?:src|href)="(https?:[^"]+)"', text):
            assert not ref.endswith((".js", ".css", ".woff2")), ref  # nothing loaded off-origin
    script = (SITE / "app.js").read_text(encoding="utf8")
    assert not re.search(r"\sstyle=|setAttribute\(\s*\"style\"", script)


def test_the_script_finds_every_element_it_draws_into() -> None:
    page = (SITE / "index.html").read_text(encoding="utf8")
    script = (SITE / "app.js").read_text(encoding="utf8")
    ids = set(re.findall(r'\$\("([\w-]+)"\)', script)) | set(
        re.findall(r'(?:frame|seg|legend)\("([\w-]+)"', script)
    )
    assert ids
    missing = sorted(i for i in ids if f'id="{i}"' not in page)
    assert not missing, missing


def test_the_diagrams_are_the_documentation_s() -> None:
    for svg in (ROOT / "docs" / "architecture").glob("*.svg"):
        served = SITE / "architecture" / svg.name
        assert served.read_bytes().replace(b"\r\n", b"\n") == svg.read_bytes().replace(
            b"\r\n", b"\n"
        )
        assert f"architecture/{svg.name}" in (SITE / "architecture.html").read_text(encoding="utf8")
