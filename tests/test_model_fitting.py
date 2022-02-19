from data import Dataset
from modelling import (
    DATA_SOURCES,
    fit_and_save_model,
    fit_models_on_all_datasets,
    load_saved_models,
    models,
    parameters,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


def test_model_fitting_and_saving(tmp_path):
    dataset = Dataset("software-reviews-sample.csv.xz")

    expected_model_destination = tmp_path / dataset.source
    expected_model_destination.mkdir(exist_ok=True)

    fit_and_save_model(
        data=dataset,
        model=models["naive_bayes"],
        param_grid=parameters["naive_bayes"],
        model_destination=tmp_path,
    )
    expected_saved_model = expected_model_destination / "MultinomialNB.xz"
    assert expected_saved_model.exists()


def test_model_fitting_and_loading(tmp_path):
    fit_models_on_all_datasets(model_destination=tmp_path)

    # Test model saving
    for source in DATA_SOURCES:
        vectorizer = tmp_path / source / "vectorizer.xz"
        assert vectorizer.exists()

        for classifier in ["LogisticRegression", "MultinomialNB", "SVC"]:
            model = tmp_path / source / f"{classifier}.xz"
            assert model.exists()

    # Test model loading
    software_reviews_models = load_saved_models(tmp_path / "software-reviews")
    assert set(software_reviews_models.keys()) == {
        "LogisticRegression",
        "MultinomialNB",
        "SVC",
        "vectorizer",
    }
    assert isinstance(
        software_reviews_models["LogisticRegression"], LogisticRegression
    )
    assert isinstance(software_reviews_models["MultinomialNB"], MultinomialNB)
    assert isinstance(software_reviews_models["SVC"], SVC)
    assert isinstance(software_reviews_models["vectorizer"], TfidfVectorizer)
