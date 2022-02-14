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

from data import Dataset

logging.basicConfig(
    format="[%(levelname)s %(asctime)s.%(msecs)03d] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)

DATA_SOURCES = ["amazon_cells", "imdb", "yelp"]
MODEL_DIR = Path("models")
SEED = 12345


models = dict(
    logistic_regression=LogisticRegression(
        class_weight="balanced", max_iter=350, random_state=SEED
    ),
    naive_bayes=MultinomialNB(),
    SVC=SVC(class_weight="balanced", random_state=SEED),
)

parameters = dict(
    logistic_regression={"C": np.logspace(-1, 4, 10)},
    naive_bayes={"alpha": np.logspace(-4, 5, 10)},
    SVC={"C": np.logspace(-1, 4, 10)},
)


def fit_and_save_model(
    data: Dataset,
    model: ClassifierMixin,
    param_grid: dict,
    model_destination: Path,
) -> None:
    model_name = type(model).__name__
    cv = StratifiedKFold(n_splits=5, shuffle=True)

    logging.info(f"Fitting {model_name} model...")
    best_model = RandomizedSearchCV(
        model, param_distributions=param_grid, cv=cv, n_jobs=-1, scoring="f1"
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

    for data_source in DATA_SOURCES:
        dataset = Dataset(f"{data_source}_labelled.txt")

        dataset_model_destination = model_destination / data_source
        dataset_model_destination.mkdir(exist_ok=True, parents=True)

        joblib.dump(
            dataset.vectorizer,
            dataset_model_destination / "vectorizer.xz",
        )

        print(
            "*" * 60
            + f"\nFitting models on the {data_source!r} dataset:\n"
            + "*" * 60
        )
        for name, model in models.items():
            fit_and_save_model(
                data=dataset,
                model=model,
                param_grid=parameters[name],
                model_destination=model_destination,
            )


if __name__ == "__main__":
    fit_models_on_all_datasets(MODEL_DIR)
