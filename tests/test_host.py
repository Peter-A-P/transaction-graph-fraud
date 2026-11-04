"""The demo page's host configuration: the committed policy allows the committed page's script
and style by hash and nothing else, so a page rebuilt without its configuration fails here
rather than as a blank page in a visitor's browser."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from consort.report import host

SITE = Path(__file__).parents[1] / "site"
TEMPLATE = Path(__file__).parents[1] / "consort" / "report" / "site.html"


def test_the_committed_configuration_matches_the_committed_page() -> None:
    page = (SITE / "index.html").read_text(encoding="utf8")
    committed = json.loads((SITE / "staticwebapp.config.json").read_text(encoding="utf8"))
    assert committed == host.config(page)


def test_the_policy_allows_only_the_inline_script_and_style() -> None:
    page = "<style>a{}</style><script>let x = 1;</script>"
    csp = host.config(page)["globalHeaders"]["Content-Security-Policy"]
    assert "default-src 'none'" in csp
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp
    changed = host.config(page.replace("1", "2"))["globalHeaders"]["Content-Security-Policy"]
    assert changed != csp  # one character of script moves the hash
    with pytest.raises(ValueError):
        host.config("<p>no script</p>")


def test_the_page_uses_nothing_the_policy_refuses() -> None:
    for text in (TEMPLATE.read_text(encoding="utf8"), (SITE / "index.html").read_text("utf8")):
        assert not re.search(r"\sstyle=", text)  # style attributes, also inside script strings
        assert not re.search(r"<script[^>]+src=|<link[^>]+stylesheet", text)
        assert not re.search(r"\son[a-z]+=", text)  # inline event handlers
