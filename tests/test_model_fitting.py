from data import Dataset
from modelling import (
    DATA_SOURCES,
    fit_and_save_model,
    fit_models_on_all_datasets,
    models,
)


def test_model_fitting_function(tmp_path):
    dataset = Dataset("yelp_labelled.txt")
    expected_model_destination = tmp_path / dataset.source
    expected_model_destination.mkdir(exist_ok=True)

    fit_and_save_model(
        data=dataset,
        model=models["naive_bayes"],
        param_grid={},
        model_destination=tmp_path,
    )
    expected_saved_model = expected_model_destination / "MultinomialNB.xz"
    assert expected_saved_model.exists()


def test_fit_and_save_all_models(tmp_path):
    fit_models_on_all_datasets(model_destination=tmp_path)
    for source in DATA_SOURCES:
        vectorizer = tmp_path / source / "vectorizer.xz"
        assert vectorizer.exists()

        for classifier in ["LogisticRegression", "MultinomialNB", "SVC"]:
            model = tmp_path / source / f"{classifier}.xz"
            assert model.exists()
