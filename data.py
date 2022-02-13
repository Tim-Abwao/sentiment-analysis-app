from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

DATA_DIR = Path("datasets")


class Dataset:
    def __init__(self, file_name, test_size=0.2) -> None:
        self.file_name = file_name
        self.source = file_name.split("_labelled.txt")[0]
        self.test_size = test_size
        self._split()

    def _load_file(self):
        return pd.read_csv(
            DATA_DIR / self.file_name,
            dtype={0: "string", 1: "int8"},
            header=None,
            names=["text", "label"],
            sep="\t",
        )

    def _vectorize_dataset(self, dataset, refit=False):
        if refit is True:
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), max_df=0.8, stop_words="english"
            )
            return self.vectorizer.fit_transform(dataset)
        else:
            return self.vectorizer.transform(dataset)

    def _split(self):
        data = self._load_file()
        (
            train_text,
            self.test_text,
            self.y_train,
            self.y_test,
        ) = train_test_split(
            data["text"],
            data["label"],
            test_size=self.test_size,
            stratify=data["label"],
        )
        self.X_train = self._vectorize_dataset(train_text, refit=True)
        self.X_test = self._vectorize_dataset(self.test_text)
