import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).parent / "src"))

from inference.py import TRTClassifier


model_engine_path: str = "models/model.engine"
label_mapping_path: str = "data/label_mapping"

@st.cache_resource
def LoadClassifier() -> TRTClassifier:
    return TRTClassifier(
        engine_path = model_engine_path,
        label_mapping_path = label_mapping_path,
    )

def main() -> None:
    st.set_page_config(
        page_title = "arXiv Paper Topic Classifier",
        layout = "centered",
    )

    st.title("arXiv Paper Topic Classifier")
    st.markdown(
        "Enter a paper title and abstract to predict the most likely topic category."
    )

    title_input: str = st.text_input(
        label = "Paper title",
        placeholder = "e.g. Attention is all you need",
    )

    abstrace_input: str = st.text_input(
        label = "(optional) Abstract",
        placeholder = "Paste your abstract here...",
        height = 150,
    )