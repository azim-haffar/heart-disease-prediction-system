# v0.1.0 — Initial Educational Release

This educational project demonstrates a reproducible evaluation workflow and a local Streamlit app. It uses a historical cohort and fictional inputs. Clinical validation and use are outside the project scope.

## Tested environment

Python 3.14.8 with NumPy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1, XGBoost 3.4.1, joblib 1.6.0, Matplotlib 3.11.2, Streamlit 1.67.0, SQLAlchemy 2.1.9, Alembic 1.20.1, psycopg and psycopg-binary 3.3.9, and pytest 9.2.0. These are the verified top-level versions. The install ranges in pyproject.toml can resolve newer versions. Use the recorded Python and model-library versions when restoring the serialized artifact.

## Running the demo

Download the source ZIP, `model.joblib`, and its checksum from the [v0.1.0 release](https://github.com/azim-haffar/heart-disease-prediction-system/releases/tag/v0.1.0).
The [demo guide](DEMO.md) covers verification and setup with the pinned Windows runtime.
Prediction uses the frozen 13-feature model and requires neither PostgreSQL nor retraining.
The public package excludes the historical reports needed by interactive Model Comparison.

## Saved evaluation

Training-only nested cross-validation mean F2 was 0.860 for Logistic Regression, 0.855 for Random Forest, 0.842 for XGBoost, and 0.809 for the always-positive baseline.

The already-consumed 61-row holdout produced 85.25% accuracy, 77.14% precision, 96.43% recall, 75.76% specificity, F1 0.8571, F2 0.9184, ROC-AUC 0.9643, average precision 0.9673, and Brier score 0.0886. The confusion matrix was [[25, 8], [1, 27]] in TN/FP and FN/TP order. The 27 detected positives and eight false positives illustrate the recall–precision trade-off on this split.

## Licensing

The original software and its documentation use [MIT](LICENSE). The UCI Cleveland
dataset retains its CC BY 4.0 attribution in [README](README.md).
Individual holdout scores and private experiment bundles are outside this release.
