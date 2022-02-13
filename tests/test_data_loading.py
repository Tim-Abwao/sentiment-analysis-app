from data import DATA_DIR, Dataset
import pandas as pd
from pandas.api.types import is_integer_dtype, is_string_dtype
from sklearn.feature_extraction.text import TfidfVectorizer
from pytest import approx

file = "yelp_labelled.txt"
test_data = pd.read_csv(
    DATA_DIR / file,
    dtype={0: "string", 1: "int8"},
    sep="\t",
    header=None,
    names=["text", "label"],
)
dataset = Dataset(file, test_size=0.3)


def test_dataset_properties():

    assert dataset.file_name == file
    assert dataset.source == "yelp"
    assert dataset.test_size == 0.3


def test_file_loading():
    expected = test_data
    actual = dataset._load_file()
    assert actual.equals(expected)
    assert is_string_dtype(actual["text"])
    assert is_integer_dtype(actual["label"])


def test_text_vectorization():
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), max_df=0.8, stop_words="english"
    )
    X_train_text = test_data.loc[dataset.y_train.index]
    vectorizer.fit_transform(X_train_text)

    sample_text = ["Just some random text"]
    expected = vectorizer.transform(sample_text)
    actual = dataset.vectorizer.transform(sample_text)
    assert actual.data == approx(expected.data)


def test_data_splitting():

    assert hasattr(dataset, "X_train")
    assert hasattr(dataset, "X_test")
    assert hasattr(dataset, "y_train")
    assert hasattr(dataset, "y_test")
    assert (dataset.y_train.shape[0] / dataset.y_test.shape[0]) == approx(
        (1 - dataset.test_size) / dataset.test_size
    )
