"""Binary evaluation and deterministic threshold selection."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    fbeta_score, precision_score, recall_score, roc_auc_score,
)

THRESHOLDS = tuple(round(value / 100, 2) for value in range(5, 96, 5))


def classify(scores, threshold):
    return (np.asarray(scores) >= threshold).astype(int)


def metrics(labels, scores, threshold=0.5):
    predicted = classify(scores, threshold)
    tn, fp, fn, tp = (int(value) for value in confusion_matrix(
        labels, predicted, labels=[0, 1],
    ).ravel())
    return {
        "accuracy": accuracy_score(labels, predicted),
        "precision": precision_score(labels, predicted, zero_division=0),
        "recall": recall_score(labels, predicted, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": fbeta_score(labels, predicted, beta=1, zero_division=0),
        "f2": fbeta_score(labels, predicted, beta=2, zero_division=0),
        "roc_auc": roc_auc_score(labels, scores),
        "average_precision": average_precision_score(labels, scores),
        "brier": brier_score_loss(labels, scores),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def threshold_search(labels, scores):
    curve = [{"threshold": t, **metrics(labels, scores, t)} for t in THRESHOLDS]
    # After F2, favor precision, then higher thresholds for repeatable ties.
    best = max(curve, key=lambda row: (row["f2"], row["precision"], row["threshold"]))
    return best["threshold"], curve
