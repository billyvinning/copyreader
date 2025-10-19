import copy
import re

import pandas as pd
import streamlit as st
from annotated_text import annotated_text

from copyreader.chain import get_chain
from copyreader.examples import _get_example_paths

example_text_path = next(_get_example_paths())
with example_text_path.open("r") as f:
    example_text = f.read()


st.set_page_config(page_title="Copyreader", page_icon="✏️")
st.header("✏️Copyreader")
st.caption("LLM text editing with human-in-the-loop.")


placeholder = st.empty()

with placeholder.container():
    text = st.text_area(
        label="Write your text here:",
        placeholder="Lorem ipsum.",
        value=example_text,
        height="content",
    )

    button_is_disabled = not bool(text)
    button_clicked = st.button(label="Submit", disabled=button_is_disabled)

if button_clicked:
    placeholder.empty()
    with placeholder, st.spinner(text="Waiting for LLM response...", show_time=True):
        if "chain" not in st.session_state:
            from langchain_google_vertexai import ChatVertexAI

            llm = ChatVertexAI(
                model="gemini-2.5-flash",
                temperature=0.2,
                max_tokens=None,
                max_retries=6,
                stop=None,
            )
            st.session_state["chain"] = get_chain(llm)
        st.session_state["response"] = st.session_state["chain"].invoke({"text": text})

if (response := st.session_state.get("response")) is not None:
    df = pd.json_normalize(response.model_dump())
    df.index.name = "Issue"
    df_t = df.rename(
        columns={
            "re_pattern": "Pattern",
            "reason": "Proposal",
            "annotation_params.repl": "Substitution",
        },
    )[["Pattern", "Proposal", "Substitution"]]

    df_t = df_t[["Pattern", "Proposal", "Substitution"]]
    with st.sidebar:
        st.title("Proposed improvements:")
        event = st.dataframe(
            df_t, hide_index=False, selection_mode="multi-row", on_select=lambda: None
        )

    augmented_text = copy.deepcopy(text)
    if event.selection.rows:
        ixs = event.selection.rows
        for _, row in df.loc[ixs, :].iterrows():
            augmented_text = re.sub(
                row["re_pattern"], row["annotation_params.repl"], augmented_text
            )
    num_annotations_applied = len(event.selection.rows)

    text_with_annotations = response.get_annotation_text(augmented_text)
    with placeholder.container():
        st.metric(
            label="No. Issues Remaining",
            value=len(response.root) - num_annotations_applied,
            delta=-num_annotations_applied,
            delta_color="inverse",
        )

        annotated_text(text_with_annotations)
