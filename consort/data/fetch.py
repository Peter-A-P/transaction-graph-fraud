"""Fetch a source's files, from Kaggle with the reader's token or from the authors' public
Google Drive folder, and verify them.

The first fetch of a source writes manifests/<source>.json (file names, sizes, SHA-256) and
that manifest is committed. Every later fetch, on any machine, is checked against it, and a
file that differs stops the build: a number in the README is only reproducible if the bytes
under it are the same bytes.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from pydantic import BaseModel

RAW = Path("data/raw")
MANIFESTS = Path("manifests")


@dataclass(frozen=True)
class Source:
    name: str
    kaggle: str  # the Kaggle slug, or "gdrive:<folder id>" for a Drive source
    files: tuple[str, ...]
    licence: str
    drive_ids: tuple[str, ...] = ()  # one Google Drive file id per file, for a Drive source


SOURCES: dict[str, Source] = {
    "elliptic": Source(
        "elliptic",
        "ellipticco/elliptic-data-set",
        (
            "elliptic_bitcoin_dataset/elliptic_txs_features.csv",
            "elliptic_bitcoin_dataset/elliptic_txs_classes.csv",
            "elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv",
        ),
        "CC BY-NC-ND 4.0",
    ),
    **{
        f"ibm-{tier}": Source(
            f"ibm-{tier}",
            "ealtman2019/ibm-transactions-for-anti-money-laundering-aml",
            (f"{tier}_Trans.csv", f"{tier}_Patterns.txt"),
            "CDLA-Sharing-1.0",
        )
        for tier in ("HI-Small", "LI-Small", "HI-Medium", "LI-Medium")
    },
    # The actor (wallet address) graph only; ids from the authors' folder, listed 2026-11-01.
    "ellipticpp": Source(
        "ellipticpp",
        "gdrive:1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l",
        ("wallets_features.csv", "wallets_classes.csv", "AddrAddr_edgelist.csv"),
        "none stated; used with both authors' written permission of 2026-11-01 (docs/data.md)",
        drive_ids=(
            "1mhrrobYdnaxYBIVK06EpKB2EwV-ERlce",
            "1ZaACVE4wSIx7r8Z9ze7ExQnJ0wzGrVkp",
            "1x5JdNWX8fVM3FO8I0wOLDAeh2qaB53uV",
        ),
    ),
}


class FileRecord(BaseModel):
    path: str
    bytes: int
    sha256: str


class Manifest(BaseModel):
    source: str
    kaggle: str
    licence: str
    first_fetched: date
    files: list[FileRecord]


class ChecksumMismatch(Exception):
    """A fetched file is not the file the committed manifest recorded."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def record(root: Path, rel: str) -> FileRecord:
    p = root / rel
    return FileRecord(path=rel, bytes=p.stat().st_size, sha256=sha256(p))


def verify(source: Source, root: Path, manifest: Manifest) -> None:
    expected = {f.path: f for f in manifest.files}
    for rel in source.files:
        if rel not in expected:
            raise ChecksumMismatch(f"{rel} is not in manifests/{source.name}.json")
        got = record(root, rel)
        if got != expected[rel]:
            raise ChecksumMismatch(
                f"{rel}: {got.bytes} bytes, sha256 {got.sha256[:12]}; "
                f"manifest has {expected[rel].bytes} bytes, {expected[rel].sha256[:12]}"
            )


def _download_drive(source: Source, root: Path) -> None:
    import gdown

    root.mkdir(parents=True, exist_ok=True)
    for rel, file_id in zip(source.files, source.drive_ids, strict=True):
        if (root / rel).exists():
            continue
        part = root / f"{rel}.part"
        # Drive drops long transfers now and then; resume from what arrived, a few times.
        for attempt in range(1, 6):
            try:
                if gdown.download(id=file_id, output=str(part), quiet=True, resume=True):
                    break
            except OSError as e:
                if attempt == 5:
                    raise
                print(f"{rel}: download interrupted ({e}); resuming")
        else:
            raise FileNotFoundError(f"Google Drive download of {rel} ({file_id}) failed")
        part.replace(root / rel)
        for stray in root.glob(f"{rel}.part*"):  # gdown's own temporary files on a resume
            stray.unlink()


def _download(source: Source, root: Path) -> None:
    if source.drive_ids:
        _download_drive(source, root)
        return
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise RuntimeError("the kaggle command is not on PATH; run inside `uv run`")
    root.mkdir(parents=True, exist_ok=True)
    for rel in source.files:
        if (root / rel).exists():
            continue
        subprocess.run(
            [
                kaggle,
                "datasets",
                "download",
                source.kaggle,
                "-f",
                rel,
                "-p",
                str(root),
                "--unzip",
                "-q",
            ],
            check=True,
        )
        # kaggle 2.x keeps the file's path inside the dataset; older versions saved the base
        # name only. Move the latter into place so both land at root / rel.
        flat = root / Path(rel).name
        if not (root / rel).exists() and flat.exists():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            flat.replace(root / rel)
        if not (root / rel).exists():
            raise FileNotFoundError(f"kaggle download did not produce {root / rel}")


def fetch(name: str, raw: Path = RAW, manifests: Path = MANIFESTS) -> Path:
    source = SOURCES[name]
    root = raw / name
    _download(source, root)
    mpath = manifests / f"{name}.json"
    if mpath.exists():
        verify(source, root, Manifest.model_validate_json(mpath.read_text()))
    else:
        manifests.mkdir(parents=True, exist_ok=True)
        m = Manifest(
            source=name,
            kaggle=source.kaggle,
            licence=source.licence,
            first_fetched=date.today(),
            files=[record(root, rel) for rel in source.files],
        )
        mpath.write_text(m.model_dump_json(indent=2) + "\n")
    return root
