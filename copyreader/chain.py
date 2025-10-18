import enum
import json
import uuid
from operator import itemgetter
from typing import Final

from langchain_core.language_models import LLM
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableAssign,
    RunnableLambda,
    RunnableParallel,
)
from pydantic import BaseModel, ConfigDict, Field, RootModel

from .split import TextChunks, _chunk_text

_SYSTEM_PROMPT: Final[str] = """\
You are a expert copyreader system. \
Your task is to help the user refine a given piece of text. \
You will be given a piece of text that has been chunked by sentence. \
The chunked piece of text will be supplied in the following JSON format: \
{chunked_text_schema}.

Adopt a friendly but professional tone. \
You are permitted to use emojis. \
Ignore empty chunks. \
Respond with your proposed edits in the following JSON format: {edits_schema}. \
"""


class ReplaceCategory(enum.StrEnum):
    "What the proposed replacement aims to improve."

    SPELLING = enum.auto()
    GRAMMAR = enum.auto()
    STYLE = enum.auto()
    CONTENT = enum.auto()


class Action(enum.StrEnum):
    "Type of action."

    REPLACE = enum.auto()
    DELETE = enum.auto()
    # INSERT = enum.auto()  # noqa: ERA001
    # MOVE = enum.auto()  # noqa: ERA001


class ReplaceActionParams(BaseModel):
    "Parameters for a proposed replacement."

    content: str = Field(description="The rewritten chunk content.")
    categories: list[ReplaceCategory]


class ProposedChunkAction(BaseModel):
    "A proposed action on a text chunk."

    id: uuid.UUID = Field(description="Identifier of the chunk.")
    action: Action = Field(description="The type of action to take.")
    reason: str = Field(description="User-facing reason for the proposed change.")

    action_params: None | ReplaceActionParams = Field(None)

    model_config = ConfigDict(use_enum_values=True)


class ProposedChunkActions(RootModel[list[ProposedChunkAction]]):
    "Some proposed actions."


def _get_prompt_template() -> ChatPromptTemplate:
    chunked_text_schema = json.dumps(TextChunks.model_json_schema())
    edits_schema = json.dumps(ProposedChunkActions.model_json_schema())

    system_message = _SYSTEM_PROMPT.format(
        chunked_text_schema=chunked_text_schema, edits_schema=edits_schema
    )
    return ChatPromptTemplate(
        [SystemMessage(system_message), ("human", "{chunked_text}")],
        input_variables=["chunked_text"],
    )


def get_chain(llm: LLM) -> Runnable:
    prompt = _get_prompt_template()
    parser = PydanticOutputParser(pydantic_object=ProposedChunkActions)
    return RunnableAssign(
        RunnableParallel(
            {"chunked_text": itemgetter("text") | RunnableLambda(_chunk_text)}
        )
    ) | RunnableAssign(RunnableParallel({"actions": prompt | llm | parser}))
