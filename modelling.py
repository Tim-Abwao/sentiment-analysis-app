import logging
from pathlib import Path
from pprint import pprint

import joblib
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from data import DATA_DIR, Dataset

logging.basicConfig(
    format="[%(levelname)s %(asctime)s.%(msecs)03d] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)

DATA_SOURCES = [
    file.name.split("-sample")[0] for file in DATA_DIR.glob("*.csv.xz")
]
MODEL_DIR = Path("models")
SEED = 12345


models = dict(
    logistic_regression=LogisticRegression(
        class_weight="balanced", solver="sag", random_state=SEED
    ),
    naive_bayes=MultinomialNB(),
    SVC=SVC(class_weight="balanced", random_state=SEED),
)

parameters = dict(
    logistic_regression={"C": np.logspace(-4, 4, 10)},
    naive_bayes={"alpha": np.logspace(-4, 4, 10)},
    SVC={"C": np.logspace(-4, 4, 10)},
)


def fit_and_save_model(
    data: Dataset,
    model: ClassifierMixin,
    param_grid: dict,
    model_destination: Path,
) -> None:
    """Fit classifier models on the provided dataset. RandomizedSearchCV is
    used to tune hyper-parameters.

    Args:
        data (data.Dataset): Dataset to model.
        model (ClassifierMixin): A scikit-learn classifier.
        param_grid (dict): Hyper-parameter values to try.
        model_destination (Path): Directory to store the fitted model.
    """
    model_name = type(model).__name__

    logging.info(f"Fitting {model_name} model...")
    best_model = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True),
        n_jobs=-1,
        scoring="f1",
    )
    best_model.fit(data.X_train, data.y_train)

    print(
        f"CV Score: {best_model.best_score_:.4f}, "
        f"Test Score: {best_model.score(data.X_test, data.y_test):.4f}, "
        f"Best Parameters:"
    )
    pprint(best_model.best_params_)
    print()  # Add newline

    joblib.dump(
        best_model.best_estimator_,
        model_destination / data.source / f"{model_name}.xz",
    )


def fit_models_on_all_datasets(model_destination: Path) -> None:
    """Fit and save models for all data sources.

    Args:
        model_destination (Path): The parent directory to store the models.
    """
    for data_source in DATA_SOURCES:
        dataset = Dataset(f"{data_source}-sample.csv.xz")
        dataset_model_destination = model_destination / data_source

        # Ensure model destination exists
        dataset_model_destination.mkdir(exist_ok=True, parents=True)

        # Persist vectorizer. Will be needed to transform new input.
        joblib.dump(
            dataset.vectorizer,
            dataset_model_destination / "vectorizer.xz",
        )

        print(
            "*" * 60
            + f"\nFitting models on the {data_source!r} dataset:\n"
            + "*" * 60
        )
        for name, classifier in models.items():
            fit_and_save_model(
                data=dataset,
                model=classifier,
                param_grid=parameters[name],
                model_destination=model_destination,
            )


def load_saved_models(model_path: Path) -> dict:
    """Fetch the pre-trained models, and the vectorizer.

    Args:
        model_path (Path): The directory containing the models.

    Returns:
        dict: The retrieved items.
    """
    return {file.name[:-3]: joblib.load(file) for file in model_path.iterdir()}


if __name__ == "__main__":
    fit_models_on_all_datasets(MODEL_DIR)
