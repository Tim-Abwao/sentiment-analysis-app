from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

DATA_DIR = Path("datasets")


class Dataset:
    """Load, partition and vectorize a file from the sample `datasets`
    directory.

    There are currently 2 files available:

    - software-reviews-sample.csv.xz
    - video-reviews-sample.csv.xz

    To add more, please follow the instructions in the README file of the
    `datasets` directory. 

    Once loaded, the file's contents are split into a training and a
    validation set, then vectorized using the scikit-learn TfidfVectorizer.

    Args:
        file_name (str): A file having text data and sentiment labels.
        test_size (float, optional): The proportion of the data to use for
            model validation. Defaults to 0.2.
    """

    def __init__(self, file_name: str, test_size: float = 0.2) -> None:
        self.file_name = file_name
        self.source = file_name.removesuffix("-sample.csv.xz")
        self.TEST_SIZE = test_size
        self._split()
        self._vectorize_text()

    def _load_file(self) -> pd.DataFrame:
        """Get data from a text file as a pandas dataframe.

        Returns:
            pandas.DataFrame: The file's contents.
        """

        return pd.read_csv(
            DATA_DIR / self.file_name,
            dtype={0: "string", 1: "int8"},
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
            data["sentiment"],
            test_size=self.TEST_SIZE,
            stratify=data["sentiment"],
        )

    def _vectorize_text(self) -> None:
        """Vectorize features using the TfidfVectorizer."""
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), max_df=0.8, stop_words="english"
        )
        self.X_train = self.vectorizer.fit_transform(self.X_train)
        self.X_test = self.vectorizer.transform(self.X_test)
