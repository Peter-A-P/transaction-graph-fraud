from pathlib import Path

import pytest

from consort import env


def test_load_sets_missing_and_keeps_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / ".env"
    p.write_text('# comment\nA_TOKEN="abc"\nexport B_KEY=x=y\nKEEP=file\nnot a line\n')
    for k in ("A_TOKEN", "B_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("KEEP", "shell")
    assert env.load(p) == ["A_TOKEN", "B_KEY"]
    import os

    assert os.environ["A_TOKEN"] == "abc"
    assert os.environ["B_KEY"] == "x=y"
    assert os.environ["KEEP"] == "shell"


def test_load_missing_file(tmp_path: Path) -> None:
    assert env.load(tmp_path / "absent") == []
