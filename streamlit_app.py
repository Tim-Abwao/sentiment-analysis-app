import pandas as pd
import streamlit as st

from modelling import DATA_SOURCES, MODEL_DIR, load_saved_models

# Configuration
title = "Sentiment Analysis App"
st.set_page_config(page_title=title)

# Introduction
st.title(title)
st.markdown(
    "Predict the *general feeling* in a body of text using [Scikit-Learn]"
    "(https://scikit-learn.org/) models."
)
text_input = st.text_area("Text Input:", max_chars=500)


@st.experimental_memo
def get_models() -> list:
    """Fetch and cache the pre-trained models."""
    return [load_saved_models(MODEL_DIR / source) for source in DATA_SOURCES]


@st.cache
def get_predictions(models: dict, text: str) -> tuple:
    """Predict the sentiments in the supplied text.

    Args:
        models (dict): The vectorizer, and classifier models
        text (str): Text to analyse.

    Returns:
        tuple: (Words found in model vocabularies, predicted sentiments).
    """
    vectorizer = models["vectorizer"]
    transformed_text = vectorizer.transform([text])
    key_features = vectorizer.inverse_transform(transformed_text)
    predictions = pd.Series(
        {
            model_name: model.predict(transformed_text)[0]
            for model_name, model in models.items()
            if model_name != "vectorizer"
        }
    )
    return key_features, predictions


def dataframe_styler(value: str) -> str:
    """Dynamically set font style and color for the results table.

    Args:
        value (str): {"Positive", "Negative"}.

    Returns:
        str: CSS style string.
    """
    base_style = "font-family:serif;font-weight:bold;size:16px;color:"
    if value == "Positive":
        return base_style + "lime;"
    return base_style + "#f22;"


def display_predictions(models: dict, text: str) -> None:
    """Present the results as a table.

    Args:
        models (dict): The pre-trained models, and the vectorizer.
        text (str): Text to analyse.
    """
    key_features, predictions = get_predictions(models, text)
    styled_predictions = (
        predictions.to_frame(name="Predicted Sentiment")
        .replace({0: "Negative", 1: "Positive"})
        .style.applymap(dataframe_styler)
    )
    st.write(styled_predictions)
    st.write("Key Features:")
    if len(key_features[0]) == 0:
        st.write(
            "None. All input words don't exist in the model's vocabulary."
            " The predicted sentiment is arbitrary."
        )
    else:
        st.write(key_features)


all_models = get_models()

if st.button("Analyse"):
    if not text_input:
        st.warning("Please supply text to analyse.")
    else:
        st.markdown("#### Predictions:")
        for data_source, models in zip(DATA_SOURCES, all_models):
            with st.expander(f"{data_source.title()} Dataset Models:"):
                display_predictions(models, text_input)
