"""The host's configuration for the demo page, written beside it by `consort site`.

The page is one file with its script and stylesheet inline, so the content security policy
allows exactly those two by their SHA-256 hashes and nothing else: no other script, no style
attribute, no request anywhere. The hashes change whenever the page does, which is why the
configuration is written from the page rather than by hand, and why a test fails when the
committed configuration has fallen behind the committed page.
"""

from __future__ import annotations

import base64
import hashlib
import re
from typing import Any


def _hashes(page: str, tag: str) -> list[str]:
    bodies = re.findall(rf"<{tag}>(.*?)</{tag}>", page, flags=re.S)
    if not bodies:
        raise ValueError(f"no inline <{tag}> in the page")
    return [
        "'sha256-" + base64.b64encode(hashlib.sha256(b.encode("utf8")).digest()).decode() + "'"
        for b in bodies
    ]


def config(page: str) -> dict[str, Any]:
    """Azure Static Web Apps' staticwebapp.config.json for `page`, the site's index.html."""
    csp = "; ".join(
        [
            "default-src 'none'",
            "script-src " + " ".join(_hashes(page, "script")),
            "style-src " + " ".join(_hashes(page, "style")),
            "img-src 'self' data:",
            "object-src 'none'",
            "frame-ancestors 'none'",
            "base-uri 'none'",
            "form-action 'none'",
        ]
    )
    return {
        "$schema": "https://json.schemastore.org/staticwebapp.config.json",
        "globalHeaders": {
            "Content-Security-Policy": csp,
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "Permissions-Policy": "geolocation=(), camera=(), microphone=(), interest-cohort=()",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        },
        "mimeTypes": {".html": "text/html", ".txt": "text/plain", ".json": "application/json"},
        "navigationFallback": {"rewrite": "/index.html"},
        "responseOverrides": {"404": {"rewrite": "/index.html", "statusCode": 200}},
    }
