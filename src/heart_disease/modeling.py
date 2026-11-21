"""Training-only model comparison and configuration selection."""

from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from .evaluation import metrics, threshold_search
from .preprocessing import build_preprocessor
from .split import SEED

OUTER_FOLDS = 5
INNER_FOLDS = 3
F2_TOLERANCE = 0.01
F2 = make_scorer(fbeta_score, beta=2, zero_division=0)


@dataclass(frozen=True)
class Candidate:
    name: str
    classifier: object
    grid: dict


def candidates():
    from xgboost import XGBClassifier

    return [
        logistic_candidate(),
        forest_candidate(),
        Candidate(
            "xgboost",
            XGBClassifier(
                objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                random_state=SEED, n_jobs=1, subsample=0.85, colsample_bytree=0.85,
                reg_lambda=2.0,
            ),
            {
                "classifier__n_estimators": [80, 140],
                "classifier__max_depth": [2, 3],
                "classifier__learning_rate": [0.04, 0.08],
            },
        ),
        dummy_candidate(),
    ]


def logistic_candidate():
    return Candidate(
        "logistic_regression", LogisticRegression(solver="lbfgs", max_iter=2000),
        {"classifier__C": [0.1, 1.0, 10.0], "classifier__class_weight": [None, "balanced"]},
    )


def forest_candidate():
    return Candidate(
        "random_forest", RandomForestClassifier(random_state=SEED, n_jobs=1),
        {
            "classifier__n_estimators": [120, 200],
            "classifier__max_depth": [3, 6],
            "classifier__min_samples_leaf": [3, 8],
            "classifier__class_weight": [None, "balanced"],
        },
    )


def dummy_candidate():
    return Candidate("always_positive", DummyClassifier(strategy="constant", constant=1), {})


def validation_folds(labels, n_splits):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return list(cv.split(np.zeros(len(labels)), labels))


def pipeline(candidate):
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("classifier", clone(candidate.classifier)),
    ])


def configure(candidate, features, labels):
    """Tune at 0.5, then select a threshold from predictions that exclude their own row."""
    estimator = pipeline(candidate)
    if not candidate.grid:
        return estimator, {"parameters": {}, "threshold": 0.5, "search": [], "threshold_curve": []}

    folds = validation_folds(labels, INNER_FOLDS)
    search = GridSearchCV(estimator, candidate.grid, scoring=F2, cv=folds, refit=False,
                          n_jobs=1, error_score="raise")
    search.fit(features, labels)
    estimator.set_params(**search.best_params_)
    scores = cross_val_predict(estimator, features, labels, cv=folds, method="predict_proba")[:, 1]
    threshold, curve = threshold_search(labels, scores)
    results = search.cv_results_
    return estimator, {
        "parameters": search.best_params_,
        "threshold": threshold,
        "search": [
            {"parameters": params, "mean_f2_at_0_5": float(mean), "std_f2_at_0_5": float(std)}
            for params, mean, std in zip(
                results["params"], results["mean_test_score"], results["std_test_score"], strict=True,
            )
        ],
        "threshold_curve": curve,
    }


def compare(all_candidates, features, labels):
    folds = validation_folds(labels, OUTER_FOLDS)
    comparison = {}
    for candidate in all_candidates:
        rows = []
        cumulative_matrix = np.zeros((2, 2), dtype=int)
        outer_scores = np.empty(len(labels))
        for number, (fit_ids, val_ids) in enumerate(folds, 1):
            fit_x, fit_y = features.iloc[fit_ids], labels[fit_ids]
            estimator, settings = configure(candidate, fit_x, fit_y)
            estimator.fit(fit_x, fit_y)
            scores = estimator.predict_proba(features.iloc[val_ids])[:, 1]
            outer_scores[val_ids] = scores
            result = metrics(labels[val_ids], scores, settings["threshold"])
            cumulative_matrix += result["confusion_matrix"]
            rows.append({"fold": number, **result, **settings})

        metric_keys = metrics(labels, outer_scores).keys() - {"confusion_matrix"}
        comparison[candidate.name] = {
            "folds": rows,
            "summary": {
                key: {
                    "mean": float(np.mean([row[key] for row in rows])),
                    "sd": float(np.std([row[key] for row in rows], ddof=1)),
                }
                for key in sorted(metric_keys)
            },
            "confusion_matrix": cumulative_matrix.tolist(),
        }
    return comparison


def select_model(comparison):
    """Use F2 with a predefined simplicity preference for small differences."""
    preference = ("logistic_regression", "random_forest", "xgboost")
    available = [name for name in preference if name in comparison]
    best = max(comparison[name]["summary"]["f2"]["mean"] for name in available)
    return next(
        name for name in available
        if best - comparison[name]["summary"]["f2"]["mean"] <= F2_TOLERANCE
    )
