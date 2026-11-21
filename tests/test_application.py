"""Exercise schema mapping, saved reports, and the optional application surfaces."""

import os
from pathlib import Path

import numpy as np
import pytest

from heart_disease.dataset import CATEGORIES, FEATURES, NULLABLE, NUMERIC
from heart_disease.demo import OPTIONS, PROFILES
from heart_disease.inference import load_trusted_model, validated_frame
from heart_disease.saved_run import verify_run

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/milestone-2"
DATA = ROOT / "data/processed.cleveland.data"
MANIFEST = ROOT / "data/split.json"


@pytest.fixture
def recorded():
    if not (RUN / "model.joblib").is_file():
        pytest.skip("Trusted historical run is available from the original workspace")
    return RUN


def test_fictional_profiles_and_categorical_labels():
    for name, (_, choices) in OPTIONS.items():
        assert set(choices) == set(CATEGORIES[name]) | ({None} if name in NULLABLE else set())
    for profile in PROFILES.values():
        assert list(validated_frame(profile).columns) == list(FEATURES)
    with pytest.raises(ValueError, match="trestbps"):
        validated_frame({**next(iter(PROFILES.values())), "trestbps": 0})


def test_saved_run_and_training_statistics(recorded):
    selection, validation, holdout, records = verify_run(recorded, DATA, MANIFEST)
    model = load_trusted_model(recorded / "model.joblib")
    assert model.metadata == selection
    classifier = model.pipeline.named_steps["classifier"]
    for key, value in selection["parameters"].items():
        assert classifier.get_params()[key.split("__", 1)[1]] == value
    preprocess = model.pipeline.named_steps["preprocess"]
    for feature, actual in zip(
        NUMERIC, preprocess.named_transformers_["numeric"].named_steps["impute"].statistics_,
        strict=True,
    ):
        position = FEATURES.index(feature)
        expected = np.median([records[i - 1][position] for i in selection["training_row_ids"]])
        assert actual == expected
    assert not set(selection["training_row_ids"]) & set(holdout["row_ids"])
    assert len(validation["outer_comparison"]) == 4
    for profile in PROFILES.values():
        result = model.predict(profile)
        assert result["classification"] == int(result["score"] >= selection["threshold"])


def test_streamlit_sections_and_prediction_without_database(recorded, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("HEART_RUN_DIR", str(recorded))
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://127.0.0.1:1/unused")
    app = testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()
    assert not app.exception
    model = load_trusted_model(recorded / "model.joblib")
    for name, profile in PROFILES.items():
        app.selectbox[0].select(name).run()
        assert len(app.number_input) + len(app.selectbox) - 1 == len(FEATURES)
        app.button[0].click().run()
        result = model.predict(profile)
        label = "Positive" if result["classification"] else "Negative"
        assert not app.exception
        assert app.subheader[0].value == f"{label} for the dataset-defined target"
        assert app.metric[0].value == f"{result['score']:.3f}"
        assert any("Decision threshold 0.30" in item.value for item in app.caption)
    app.radio[0].set_value("Model Comparison").run()
    assert not app.exception and len(app.dataframe) == 1
    assert len(app.dataframe[0].value) == 4
    assert "F1" in app.dataframe[0].value
    assert sum(node.type in {"image", "imgs"} for node in app) == 3
    assert any("database is unavailable" in item.value for item in app.caption)
    app.radio[0].set_value("Dataset and Methodology").run()
    assert not app.exception
    assert any("clinical diagnosis" in item.value for item in app.caption)


def test_streamlit_rejects_inconsistent_catalog(recorded, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    pytest.importorskip("sqlalchemy")
    from heart_disease import tracking

    monkeypatch.setenv("HEART_RUN_DIR", str(recorded))
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://127.0.0.1:1/unused")
    monkeypatch.setattr(tracking, "read_results", lambda *_: {
        ("holdout", "logistic_regression"): {"accuracy": 1.0},
    })
    app = testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()
    app.radio[0].set_value("Model Comparison").run()
    assert not app.exception
    assert any("catalog disagrees" in item.value for item in app.caption)
    assert app.metric[0].value == "85.2%"


def test_streamlit_missing_model(tmp_path, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("HEART_RUN_DIR", str(tmp_path))
    app = testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()
    assert not app.exception
    assert any("missing" in warning.value for warning in app.warning)


def test_postgres_migration_registration_and_retrieval(recorded, monkeypatch):
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL must point to a disposable PostgreSQL database")
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import event, select
    from sqlalchemy.orm import Session
    from heart_disease import tracking

    monkeypatch.setenv("DATABASE_URL", url)
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
    engine = tracking.engine_from_url(url)
    try:
        key, _ = tracking.register(engine, recorded, DATA, MANIFEST)
        assert (key, False) == tracking.register(engine, recorded, DATA, MANIFEST)
        selection, validation, holdout, _ = verify_run(recorded, DATA, MANIFEST)
        expected = {("validation", name): result["summary"]
                    for name, result in validation["outer_comparison"].items()}
        expected["holdout", selection["algorithm"]] = holdout["metrics"]
        assert tracking.read_results(engine, key) == expected
        with Session(engine) as session:
            assert session.get(tracking.ModelFile, key).checksum == tracking.digest(recorded / "model.joblib")

        rollback_key = "0" * 64
        monkeypatch.setattr(tracking, "fingerprint", lambda _: rollback_key)

        def block_flush(*args):
            raise RuntimeError("Simulated persistence failure")

        event.listen(Session, "before_flush", block_flush)
        try:
            with pytest.raises(RuntimeError, match="Simulated persistence failure"):
                tracking.register(engine, recorded, DATA, MANIFEST)
        finally:
            event.remove(Session, "before_flush", block_flush)
        with Session(engine) as session:
            assert session.scalar(select(tracking.TrainingRun).filter_by(fingerprint=rollback_key)) is None
    finally:
        engine.dispose()
