"""Importing from ``baselines/`` without modifying it and without cross-contamination.

``baselines/mar``, ``baselines/CSpD``, and ``baselines/djepa-imagenet`` all ship a top-level package
literally named ``models``, plus ``util`` and ``diffusion``, and all use absolute imports
(``from models.diffloss import DiffLoss``). Importing two of them in one process without care gives
the second one the first one's modules, so a CSpD run would silently execute MAR's denoising head.
:func:`baseline_imports` isolates each import so that cannot happen.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Module names more than one baseline defines. Purged around every baseline import.
COLLIDING = ("models", "util", "diffusion", "engine", "misc", "spec_decode")


def baseline_path(name: str) -> Path:
    path = REPO_ROOT / "baselines" / name
    if not path.exists():
        raise FileNotFoundError(f"baseline repo {path} not found; run scripts/setup_baselines.sh")
    return path


@contextmanager
def baseline_imports(repo: str | Path) -> Iterator[None]:
    """Temporarily make ``repo`` importable, hiding colliding modules already loaded.

    Accepts either a path or a bare baseline name (``"mar"``, ``"CSpD"``). On exit the previous
    ``sys.path`` and ``sys.modules`` are restored, so a later import of a different baseline starts
    clean.
    """
    repo = Path(repo) if Path(repo).is_absolute() or Path(repo).exists() else baseline_path(str(repo))
    if not repo.exists():
        raise FileNotFoundError(f"baseline repo {repo} not found; run scripts/setup_baselines.sh")

    saved: dict[str, object] = {}
    for name in list(sys.modules):
        if name.split(".")[0] in COLLIDING:
            saved[name] = sys.modules.pop(name)
    sys.path.insert(0, str(repo))

    # Importing writes __pycache__ into the baseline, which leaves `git -C baselines/<repo> status`
    # permanently dirty and destroys its value as a "nobody edited upstream" check.
    wrote_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = wrote_bytecode
        if sys.path and sys.path[0] == str(repo):
            sys.path.pop(0)
        for name in list(sys.modules):
            if name.split(".")[0] in COLLIDING:
                del sys.modules[name]
        sys.modules.update(saved)
