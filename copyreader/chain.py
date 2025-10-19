# ruff: noqa: E501
import enum
import re
from collections.abc import Generator
from typing import Final

import numpy as np
import numpy.typing as npt
from langchain_core.language_models import LLM
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
)
from pydantic import BaseModel, Field, RootModel

_SYSTEM_PROMPT: Final[str] = """\
You are a expert copyreader system. \
Your task is to help the user improve and refine a given piece of text. \
The given text may come in the form of Markdown, LaTeX, YAML, JSON, plain text and so on. \
Your proposed improvements should come in the form of annotations, where an annotation is a proposed change to a specific snippet of text. \
Your annotations will be fed back to the user, who will then decided whether to apply the improvements.

Rules:
1) Adopt a friendly but professional tone. \
2) You are permitted to use emojis. \
3) Don't suggest factual changes to the content that you are unsure of. \
4) Don't hallucinate. \
5) Your regex patterns will be used to match all occurrences in the text. Therefore, write your patterns in such a way to avoid matching on unwanted occurrences in the text. For example, if you intended to replace "that is" with "that are", using the pattern "is" would erroneously match the "is" in the word "history". Therefore, you must make sure your regex patterns are accurate.

{format_instructions}\
"""


class AnnotationKind(enum.StrEnum):
    "Type of annotation."

    SUBSTITUTE = enum.auto()
    DELETE = enum.auto()
    # INSERT = enum.auto()  # noqa: ERA001
    # MOVE = enum.auto()  # noqa: ERA001


class SubstitutionKind(enum.StrEnum):
    "Motivation for the replacement."

    SPELLING = enum.auto()
    GRAMMAR = enum.auto()
    STYLE = enum.auto()
    CONTENT = enum.auto()
    PUNCTUATION = enum.auto()


class SubstitutionParameters(BaseModel):
    "Parameters for a proposed substitution as in `import re; re.sub(...)`."

    repl: str = Field(description="The replacement string.")
    count: int = Field(
        default=0,
        description=(
            "The maximum number of pattern occurrences to be replaced; "
            "must be a non-negative integer. If zero, all occurrences "
            "will be replaced."
        ),
    )

    kind: list[SubstitutionKind] = Field(
        description="What the proposed substitution addresses."
    )


class Annotation(BaseModel):
    "An annotation to be applied."

    re_pattern: str = Field(
        description=(
            "A regular expression pattern to match occurrences to be annotated."
        ),
        examples=[r"foo", r"def\s+([a-zA-Z_][a-zA-Z_0-9]*)\s*\(\s*\):"],
    )
    re_flag: int = Field(
        default=0,
        description=(
            "Regular expression flag used along with the pattern to match occurrences "
            "to be annotated."
        ),
    )

    kind: AnnotationKind = Field(
        description="Type of annotation.",
    )
    reason: str = Field(description="User-facing reason for the annotation.")

    annotation_params: None | SubstitutionParameters = Field(
        default=None,
        description="Extra parameters according to the type of annotation.",
    )

    def _iter_annotation_spans(
        self, text: str
    ) -> Generator[tuple[int, int], None, None]:
        for match in re.finditer(self.re_pattern, text, self.re_flag):
            yield match.span()

    def get_annotation_mask(self, text: str) -> npt.NDArray[np.bool_]:
        spans = self._iter_annotation_spans(text)
        mask = np.zeros(len(text), dtype=bool)
        for lower, upper in spans:
            mask[lower:upper] = True
        return mask


class Annotations(RootModel[list[Annotation]]):
    "Annotations to be applied."

    def _get_annotation_masks(
        self, text: str
    ) -> dict[str | None, npt.NDArray[np.bool_]]:
        out: dict[str | None, npt.NDArray[np.bool_]] = {}
        for annotation in self.root:
            mask = annotation.get_annotation_mask(text)

            if annotation.kind == AnnotationKind.SUBSTITUTE and isinstance(
                annotation.annotation_params, SubstitutionParameters
            ):
                scopes = ",".join(annotation.annotation_params.kind)
                annotation_key = f"{annotation.kind}({scopes})"
            else:
                annotation_key = str(annotation.kind)

            if annotation_key in out:
                out[annotation_key] |= mask
            else:
                out[annotation_key] = mask

        null_mask = np.ones(len(text), dtype=bool)
        for mask in out.values():
            null_mask ^= mask
        out[None] = null_mask
        return out

    def _get_annotation_text(self, text: str) -> list[str | tuple[str, str]]:
        masks = self._get_annotation_masks(text)
        ix_to_annotation_mapping = dict(enumerate(masks))

        one_hot_masks = np.column_stack(list(masks.values()))
        dense_masks = np.argmax(one_hot_masks, axis=1)
        n = len(dense_masks)

        out = []
        i = 0
        while i < n:
            j = i + 1
            while j < n and dense_masks[i] == dense_masks[j]:
                j += 1

            annotation = ix_to_annotation_mapping[dense_masks[i]]
            item = text[i:j] if annotation is None else (text[i:j], str(annotation))
            out.append(item)
            i = j
        return out


def get_chain(llm: LLM) -> Runnable:
    parser = PydanticOutputParser(pydantic_object=Annotations)
    system_message = _SYSTEM_PROMPT.format(
        format_instructions=parser.get_format_instructions()
    )
    prompt = ChatPromptTemplate(
        [SystemMessage(system_message), ("human", "{text}")],
    )
    return prompt | llm | parser
