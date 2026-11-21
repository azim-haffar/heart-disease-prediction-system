"""Build an unfitted transformer and partition Cleveland records."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .dataset import CATEGORIES, FEATURES, NUMERIC, Record


def training_features(records: list[Record], train_ids: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        [records[row_id - 1][:-1] for row_id in train_ids],
        columns=list(FEATURES),
        index=pd.Index(train_ids, name="source_row"),
        dtype=np.float64,
    )


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encode", OneHotEncoder(
            categories=list(CATEGORIES.values()),
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])
    return ColumnTransformer([
        ("numeric", numeric, list(NUMERIC)),
        ("categorical", categorical, list(CATEGORIES)),
    ])
