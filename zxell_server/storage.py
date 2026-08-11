import hashlib
from pathlib import Path

from config import settings


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def shards_dir() -> Path:
    return _ensure(settings.storage_dir / "shards")


def weights_dir() -> Path:
    return _ensure(settings.storage_dir / "weights")


def results_dir() -> Path:
    return _ensure(settings.storage_dir / "results")


def safe_name(name: str) -> str:
    """パストラバーサル対策: 単一のファイル名成分のみ許可する。"""
    base = Path(name).name
    if base != name or base in ("", ".", ".."):
        raise ValueError("invalid file name: %r" % name)
    return base


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
