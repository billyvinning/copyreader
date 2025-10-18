from pathlib import Path

import pytest

from copyreader.examples import _get_example_paths
from copyreader.split import _split_text

_EXAMPLE_PATHS: list[Path] = list(_get_example_paths())


@pytest.mark.parametrize("path", _EXAMPLE_PATHS)
def test_split_text(path: Path) -> None:
    with path.open("r") as f:
        text = f.read()
    chunks = _split_text(text)
    assert text == "".join(chunks)
