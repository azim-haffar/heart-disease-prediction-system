from pathlib import Path

import numpy as np
from sklearn.exceptions import NotFittedError
import pytest

from heart_disease.dataset import CATEGORIES, NUMERIC, load_snapshot
from heart_disease.preprocessing import build_preprocessor, training_features
from heart_disease.split import split_rows

SNAPSHOT = Path(__file__).resolve().parents[1] / "data" / "processed.cleveland.data"


def test_preprocessor_fits_training_statistics_only():
    records, _ = load_snapshot(SNAPSHOT)
    train_ids, _ = split_rows(records)
    train = training_features(records, train_ids)
    train.loc[train_ids[0], "age"] = np.nan
    probe = train.iloc[:1].copy()
    probe["age"] = 10000.0
    probe["thal"] = 99.0
    transformer = build_preprocessor()

    with pytest.raises(NotFittedError):
        transformer.transform(probe)
    transformed = transformer.fit_transform(train)
    expected_columns = len(NUMERIC) + sum(map(len, CATEGORIES.values()))
    assert transformed.shape == (242, expected_columns)
    assert np.isfinite(transformed).all()
    np.testing.assert_allclose(transformed[:, :len(NUMERIC)].mean(axis=0), 0, atol=1e-10)

    numeric = transformer.named_transformers_["numeric"]
    age_position = NUMERIC.index("age")
    assert numeric.named_steps["impute"].statistics_[age_position] == train["age"].median()
    categorical = transformer.named_transformers_["categorical"]
    thal_position = list(CATEGORIES).index("thal")
    assert categorical.named_steps["impute"].statistics_[thal_position] == train["thal"].mode()[0]
    probe_features = transformer.transform(probe)
    assert np.isfinite(probe_features).all()
    assert (probe_features[0, -len(CATEGORIES["thal"]):] == 0).all()

    missing_thal = probe.assign(thal=np.nan)
    mode_thal = probe.assign(thal=train["thal"].mode()[0])
    np.testing.assert_array_equal(transformer.transform(missing_thal), transformer.transform(mode_thal))
