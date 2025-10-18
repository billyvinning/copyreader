import uuid
from collections.abc import Generator
from typing import Final

from pydantic import BaseModel, Field, RootModel

_SPLIT_CHARS: Final[tuple[str, ...]] = (".", "\n", "\n\n")  # , "\uff0e", "\u3002")


class TextChunk(BaseModel):
    "Representation of a chunk of text."

    id: uuid.UUID = Field(description="Unique identifier of the chunk.")
    content: str = Field(description="Text content of the chunk.")


class TextChunks(RootModel[list[TextChunk]]):
    "Representation of many text chunks."


def _splitkeep(s: str, delimiter: str) -> list[str]:
    split = s.split(delimiter)
    return [substr + delimiter for substr in split[:-1]] + [split[-1]]


def _split_text(
    s: str, remaining_split_chars: list[str] | None = None
) -> Generator[str, None, None]:
    if remaining_split_chars is None:
        remaining_split_chars = list(_SPLIT_CHARS)
    try:
        split_char = remaining_split_chars.pop()
    except IndexError:
        yield s
    else:
        for chunk in _splitkeep(s, split_char):
            yield from _split_text(chunk, remaining_split_chars[:])


def _chunk_text(s: str) -> TextChunks:
    text_chunks = []
    for chunk in _split_text(s):
        text_chunk = TextChunk(id=uuid.uuid4(), content=chunk)
        text_chunks.append(text_chunk)
    return TextChunks(text_chunks)
