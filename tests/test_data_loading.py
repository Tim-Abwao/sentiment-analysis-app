import pandas as pd
from data import DATA_DIR, Dataset
from pandas.api.types import is_integer_dtype, is_string_dtype
from pytest import approx
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

file = "software-reviews-sample.csv.xz"
test_data = pd.read_csv(
    DATA_DIR / file,
    dtype={0: "string", 1: "int8"},
)
dataset = Dataset(file, test_size=0.3)


def test_dataset_general_properties():
    assert dataset.file_name == file
    assert dataset.source == "software-reviews"
    assert dataset.TEST_SIZE == 0.3


def test_file_loading():
    expected = test_data
    actual = dataset._load_file()
    assert actual.equals(expected)
    assert is_string_dtype(actual["text"])
    assert is_integer_dtype(actual["sentiment"])


def test_text_vectorization():
    # The document-term matrix (vectorized text) should be a csr_matrix
    assert isinstance(dataset.X_train, csr_matrix)
    assert isinstance(dataset.X_test, csr_matrix)


def test_data_splitting():
    assert hasattr(dataset, "X_train")
    assert hasattr(dataset, "X_test")
    assert hasattr(dataset, "y_train")
    assert hasattr(dataset, "y_test")
    assert len(dataset.y_test) / (
        len(dataset.y_train) + len(dataset.y_test)
    ) == approx(0.3)
