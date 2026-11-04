"""The host's configuration for the demo site, written beside it by `consort site`.

The site is a page, a stylesheet, a script, two fonts, the diagrams and data.json, all served
from the one origin, so the content security policy allows this origin and nothing else: no
inline script, no style attribute, no request to anywhere else. tests/test_host.py fails if the
committed configuration differs from this one, or if the page uses anything it refuses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CSP = "; ".join(
    [
        "default-src 'none'",
        "script-src 'self'",
        "style-src 'self'",
        "img-src 'self' data:",
        "font-src 'self'",
        "media-src 'self'",
        "connect-src 'self'",
        "object-src 'none'",
        "frame-ancestors 'none'",
        "base-uri 'none'",
        "form-action 'none'",
    ]
)


def config() -> dict[str, Any]:
    """Azure Static Web Apps' staticwebapp.config.json for the site."""
    return {
        "$schema": "https://json.schemastore.org/staticwebapp.config.json",
        "globalHeaders": {
            "Content-Security-Policy": CSP,
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "Permissions-Policy": "geolocation=(), camera=(), microphone=(), interest-cohort=()",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        },
        "mimeTypes": {
            ".json": "application/json",
            ".js": "text/javascript",
            ".css": "text/css",
            ".html": "text/html",
            ".woff2": "font/woff2",
            ".txt": "text/plain",
            ".svg": "image/svg+xml",
            ".mp4": "video/mp4",
            ".jpg": "image/jpeg",
        },
        "routes": [
            {
                "route": "/data.json",
                "headers": {"Cache-Control": "public, max-age=600, must-revalidate"},
            }
        ],
        "navigationFallback": {
            "rewrite": "/index.html",
            "exclude": ["/data.json", "/*.css", "/*.js", "/fonts/*", "/architecture/*", "/video/*"],
        },
        "responseOverrides": {"404": {"rewrite": "/index.html", "statusCode": 200}},
    }


def write(out: Path) -> None:
    """The configuration and robots.txt, into the site folder."""
    (out / "staticwebapp.config.json").write_text(
        json.dumps(config(), indent=2) + "\n", encoding="utf8"
    )
    (out / "robots.txt").write_text("User-agent: *\nAllow: /\n", encoding="utf8")
