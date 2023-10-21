import pandas as pd
import streamlit as st

from modelling import DATA_SOURCES, MODEL_DIR, load_saved_models

title = "Sentiment Analysis App"
st.set_page_config(page_title=title)
st.title(title)
st.markdown(
    "Predict the *general feeling* in a body of text using the traditional "
    "[*Bag of Words*](https://en.wikipedia.org/wiki/Bag-of-words_model) "
    "model, implemented with [Scikit-Learn](https://scikit-learn.org/)."
)
with st.form("text-input-form"):
    text_input = st.text_area("Text Input:", height=140, max_chars=500)
    st.form_submit_button("Analyse")


@st.cache_resource
def get_models() -> list[dict]:
    """Fetch and cache assets for all data sources.

    Returns:
        list[dict]: Each list element has the keys "vectorizer",
        "LogisticRegression", "MultinomialNB" and "SVC".
    """
    return [load_saved_models(MODEL_DIR / source) for source in DATA_SOURCES]


def get_predictions(models: dict, text: str) -> tuple:
    """Predict the sentiment in the supplied text.

    Args:
        models (dict): Fitted vectorizer and classifiers.
        text (str): Text to analyse.

    Returns:
        tuple: (predicted sentiments, ngrams used in making prediction).
    """
    models = models.copy()  # avoid mutating cached models
    vectorizer = models.pop("vectorizer")
    transformed_text = vectorizer.transform([text])
    predictions = pd.DataFrame(
        [
            (model_name, model.predict(transformed_text)[0])
            for model_name, model in models.items()
        ],
        columns=["Model", "Predicted Sentiment"],
    ).replace({0: "Negative", 1: "Positive"})
    key_features = vectorizer.inverse_transform(transformed_text)
    return predictions, key_features


def customize_dataframe_style(label: str) -> str:
    """Dynamically set font style and color for the results table.

    Args:
        label (str): {"Positive", "Negative"}.

    Returns:
        str: CSS style string.
    """
    color = "lime" if label == "Positive" else "tomato"
    return f"font-family:serif;font-weight:bold;size:16px;color:{color};"


def display_predictions(models: dict, text: str) -> None:
    """Show the results as a table.

    Args:
        models (dict): Fitted vectorizer and models.
        text (str): Text to analyse.
    """
    predictions, key_features = get_predictions(models, text)
    st.dataframe(
        predictions.style.map(
            customize_dataframe_style, subset=["Predicted Sentiment"]
        ),
        hide_index=True,
    )
    st.write("Key Features:")
    if len(key_features[0]) == 0:
        st.caption(
            ":orange[All input words don't exist in the model's vocabulary. "
            "Predicted sentiment is arbitrary.]"
        )
    else:
        st.caption(key_features[0])


all_models = get_models()
if text_input:
    st.markdown("#### Predictions:")
    for data_source, models in zip(DATA_SOURCES, all_models):
        with st.expander(f"{data_source.title()} Dataset Models:"):
            display_predictions(models, text_input)
