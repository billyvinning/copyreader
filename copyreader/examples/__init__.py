import importlib.resources
from collections.abc import Generator
from pathlib import Path

_PACKAGE = importlib.resources.files(__package__)


def _get_example_paths() -> Generator[Path, None, None]:
    return Path(str(_PACKAGE)).glob("example_*.txt")
