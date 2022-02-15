from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

DATA_DIR = Path("datasets")


class Dataset:
    """Load, partition and vectorize data from a file.

    The data is expected to contain 2 columns, with the first being the corpus
    (array of strings), and the second being a binary (2-class-only)
    sentiment label.

    The text data is vectorized using the scikit-learn TfidfVectorizer.

    Args:
        file_name (str): A file having text data and sentiment labels.
        test_size (float, optional): The proportion of the data to use for
            model validation. Defaults to 0.2.
    """

    def __init__(self, file_name: str, test_size: float = 0.2) -> None:
        self.file_name = file_name
        self.source = file_name.split("_labelled.txt")[0]
        self.test_size = test_size
        self._split()
        self._vectorize_text()

    def _load_file(self, sep: str = "\t") -> pd.DataFrame:
        """Get data from a text file as a pandas dataframe.

        Args:
            sep (str, optional): The delimiter in the file. Defaults to "\t".

        Returns:
            pandas.core.frame.DataFrame: The file's contents.
        """

        return pd.read_csv(
            DATA_DIR / self.file_name,
            dtype={0: "string", 1: "int8"},
            header=None,
            names=["text", "label"],
            sep=sep,
        )

    def _split(self) -> None:
        """Partition the data into training and validation sets."""
        data = self._load_file()
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = train_test_split(
            data["text"],
            data["label"],
            test_size=self.test_size,
            stratify=data["label"],
        )

    def _vectorize_text(self) -> None:
        """Vectorize features using the TfidfVectorizer."""
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), max_df=0.8, stop_words="english"
        )
        self.X_train = self.vectorizer.fit_transform(self.X_train)
        self.X_test = self.vectorizer.transform(self.X_test)
