"""Milestone 2 checks. The real final holdout is never scored by tests."""

from pathlib import Path

import joblib
import numpy as np
import pytest

from heart_disease.dataset import FEATURES, SHA256, load_snapshot
from heart_disease.evaluation import classify, metrics, threshold_search
from heart_disease.experiment import dataset_rows, frozen_split
from heart_disease.inference import MODEL_VERSION, ModelArtifact, input_schema, load_trusted_model
from heart_disease.modeling import (
    OUTER_FOLDS, candidates, compare, configure, dummy_candidate, forest_candidate,
    logistic_candidate, select_model, validation_folds,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def training():
    records, checksum = load_snapshot(ROOT / "data/processed.cleveland.data")
    train, test, _ = frozen_split(ROOT / "data/split.json", records, checksum)
    return (*dataset_rows(records, train), test)


def test_threshold_and_metrics():
    labels, scores = [0, 0, 1, 1], [0.1, 0.7, 0.4, 0.9]
    assert list(classify(scores, 0.4)) == [0, 1, 1, 1]
    result = metrics(labels, scores, 0.4)
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    assert result["recall"] == 1
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["f2"] == pytest.approx(10 / 11)
    threshold, _ = threshold_search(labels, scores)
    assert threshold == 0.4


def test_folds_are_reproducible_and_exclude_holdout(training):
    features, labels, holdout = training
    folds = validation_folds(labels, OUTER_FOLDS)
    repeat = validation_folds(labels, OUTER_FOLDS)
    validation_ids = []
    for (fit, val), (next_fit, next_val) in zip(folds, repeat, strict=True):
        np.testing.assert_array_equal(fit, next_fit)
        np.testing.assert_array_equal(val, next_val)
        assert not set(fit) & set(val)
        assert not set(features.index[fit]) & set(holdout)
        assert not set(features.index[val]) & set(holdout)
        assert set(labels[val]) == {0, 1}
        validation_ids.extend(val)
    assert sorted(validation_ids) == list(range(242))


def test_selection_is_deterministic_and_dummy_is_ineligible():
    def summary(f2):
        return {"summary": {"f2": {"mean": f2}}}

    comparison = {
        "logistic_regression": summary(0.855), "random_forest": summary(0.861),
        "xgboost": summary(0.862), "always_positive": summary(0.99),
    }
    assert select_model(comparison) == "logistic_regression"
    comparison["xgboost"] = summary(0.90)
    assert select_model(comparison) == "xgboost"


@pytest.mark.parametrize("candidate", [logistic_candidate(), forest_candidate(), dummy_candidate()],
                         ids=lambda candidate: candidate.name)
def test_candidate_pipeline_schema_and_fitting(training, candidate):
    features, labels, _ = training
    configured, settings = configure(candidate, features, labels)
    configured.fit(features, labels)
    assert list(configured.feature_names_in_) == list(FEATURES)
    assert configured.named_steps["preprocess"].transform(features).shape == (242, 28)
    assert np.isfinite(configured.predict_proba(features)).all()
    if candidate.name == "always_positive":
        result = metrics(labels, configured.predict_proba(features)[:, 1], settings["threshold"])
        assert (result["recall"], result["specificity"], result["fp"]) == (1, 0, 131)


def test_xgboost_pipeline(training):
    pytest.importorskip("xgboost")
    features, labels, _ = training
    candidate = next(item for item in candidates() if item.name == "xgboost")
    estimator, configuration = configure(candidate, features, labels)
    estimator.fit(features, labels)
    assert list(estimator.feature_names_in_) == list(FEATURES)
    assert configuration["parameters"]
    assert np.isfinite(estimator.predict_proba(features)).all()


def test_reproducible_configuration_and_inference_roundtrip(training, tmp_path):
    features, labels, _ = training
    candidate = logistic_candidate()
    estimator, first = configure(candidate, features, labels)
    _, second = configure(candidate, features, labels)
    assert first == second
    estimator.fit(features, labels)
    metadata = {
        "model_version": MODEL_VERSION, "dataset_sha256": SHA256, "features": list(FEATURES),
        "input_schema": input_schema(), "threshold": first["threshold"],
    }
    artifact = ModelArtifact(estimator, metadata)
    path = tmp_path / "model.joblib"
    joblib.dump(artifact, path)
    reloaded = load_trusted_model(path)
    row = features.iloc[0].to_dict()
    assert reloaded.predict(row) == artifact.predict(row)
    assert reloaded.predict(dict(reversed(list(row.items())))) == artifact.predict(row)
    assert 0 <= reloaded.predict({**row, "ca": None})["score"] <= 1
    for bad in ({**row, "thal": 99}, {**row, "age": float("nan")}, {**row, "sex": True},
                {**row, "age": "63"}, {**row, "extra": 1}, {}):
        with pytest.raises(ValueError):
            reloaded.predict(bad)


def test_nested_dummy_control_has_no_false_negatives(training):
    features, labels, _ = training
    results = compare([dummy_candidate()], features, labels)["always_positive"]
    assert results["confusion_matrix"] == [[0, 131], [0, 111]]
    assert results["summary"]["recall"]["mean"] == 1
    assert results["summary"]["f2"]["mean"] < 1

def test_reporting_uses_synthetic_results(tmp_path):
    pytest.importorskip("matplotlib")
    from heart_disease.reporting import save_report

    labels, scores = [0, 0, 1, 1], [0.1, 0.7, 0.4, 0.9]
    threshold, curve = threshold_search(labels, scores)
    result = metrics(labels, scores, threshold)
    names = ("logistic_regression", "random_forest", "xgboost")
    configs = {name: {"threshold": threshold, "parameters": {}, "threshold_curve": curve}
               for name in names}
    comparison = {
        name: {"summary": {key: {"mean": val, "sd": 0.0}
                           for key, val in result.items() if key != "confusion_matrix"}}
        for name in names
    }
    save_report(tmp_path, {"algorithm": names[0], "threshold": threshold}, comparison, configs,
                result, [{"source_row": 2, "prediction": 1}], labels, scores)
    assert (tmp_path / "evaluation.md").is_file()
    for name in ("thresholds", "confusion-matrix", "holdout-curves"):
        assert (tmp_path / f"{name}.png").read_bytes().startswith(b"\x89PNG")
