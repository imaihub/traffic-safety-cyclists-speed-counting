"""Information about the computer, shown in the GUI start page and by ``cyclist-tracker check``."""

from __future__ import annotations

import importlib.util
import platform
import sys

from . import __version__


def system_report() -> list[tuple[str, str, bool]]:
    """Rows of (item, value, ok)."""
    rows = [
        ("Cyclist Tracker", __version__, True),
        ("Python", sys.version.split()[0], sys.version_info >= (3, 11)),
        ("Operating system", f"{platform.system()} {platform.release()}", True),
    ]
    try:
        import torch

        rows.append(("PyTorch", torch.__version__, True))
        if torch.cuda.is_available():
            memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            rows.append(("GPU", f"{torch.cuda.get_device_name(0)} ({memory:.0f} GB)", True))
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            rows.append(("GPU", "Apple Metal (MPS)", True))
        else:
            rows.append(("GPU", "not found - everything works, but analysis and training are slow", False))
    except ImportError:
        rows.append(("PyTorch", "not installed - run 'uv sync'", False))
    for label, module, hint in (
        ("OSNet feature extractor", "torchreid", "optional: uv sync --extra torchreid"),
        ("CLIP feature extractor", "clip", "optional: uv sync --extra clip"),
    ):
        found = importlib.util.find_spec(module) is not None
        rows.append((label, "installed" if found else hint, found))
    return rows
