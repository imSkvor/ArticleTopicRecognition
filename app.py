import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).parent / "src"))

from inference import TRTClassifier


model_engine_path: str = "models/model.engine"
label_mapping_path: str = "data/label_mapping.json"
checkpoints_path: str = "output/checkpoints"

@st.cache_resource
def LoadClassifier() -> TRTClassifier:
    return TRTClassifier(
        engine_path = model_engine_path,
        label_mapping_path = label_mapping_path,
        checkpoint_dir = checkpoints_path,
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

    abstract_input: str = st.text_input(
        label = "(optional) Abstract",
        placeholder = "Paste your abstract here...",
    )

    if st.button(
        label = "Classify",
        type = "primary",
        use_container_width = True,
        disabled = not (title_input.strip() or abstract_input.strip())
    ):
        if not title_input.strip() and not abstract_input.strip():
            st.error("Please enter a least a title or an abstract")
            return

        with st.spinner("Running inference with TensorRT..."):
            try:
                classifier: TRTClassifier = LoadClassifier()
                predictions: list[tuple[str, float]] = classifier.predict(
                    title = title_input,
                    abstract = abstract_input,
                    cumulative_threshold = 0.95,
                )

                st.success("Classification complete")
                st.subheader("Predicted topics (top 95% cumulative probability)")

                for rank, item in enumerate(predictions):
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(f"**{rank + 1}. {item["display"]}**")
                    
                    with col2:
                        st.metric(
                            label = "Probability",
                            value = f"{item["probability"]:.2%}",
                        )
                    
                    st.progress(item["probability"])

            except Exception as e:
                st.error(f"Error during inference: {str(e)}")
                st.exception(e)

    with st.sidebar:
        st.header("Info")
        st.markdown(
            "This service classifies research papers into arXiv topics"
            "using a BERT model running with TensorRT"
        )
        
        st.divider()

        st.subheader("Performance")
        st.markdown(
            "- **Model**: BERT-base-cased\n"
            "- **Backend**: TensorRT\n"
            "- **Precision**: FP16\n"
            # "- **Avg latency: **"
        )

        st.divider()
        st.caption("Build with streamlit + TensorRT")
        st.caption(
            "Code and description are avalible on\n"
            "https://github.com/imSkvor/ArticleTopicRecognition"
        )

if __name__ == "__main__":
    main()