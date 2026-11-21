# Milestone 2 verification

**HOLD — EXTERNAL BLOCKER**

The Windows project environment ran Python 3.14.8, NumPy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1, joblib 1.6.0, and pytest 9.2.0. No new packages were installed. pip could not find distributions for XGBoost or Matplotlib, and direct shell requests to both PyPI and its wheel host failed DNS resolution. The experiment command stopped at the Matplotlib import, before creating its output directory. The XGBoost pipeline test separately confirmed XGBoost was unavailable.

## Executed training-only comparison

Each available model used the same five outer folds within the 242 training rows. Logistic regression and random forest also used the same three-way inner split for tuning and threshold selection. Metrics are mean ± sample standard deviation across outer folds. False positives and negatives are totals across the outer validation rows.

| Model | Accuracy | Precision | Recall | Specificity | F1 | F2 | ROC-AUC | Average precision | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: |
| Logistic regression | .718 ± .085 | .636 ± .077 | .946 ± .037 | .525 ± .158 | .758 ± .056 | .860 ± .036 | .889 ± .024 | .889 ± .042 | 62 | 6 |
| Random forest | .690 ± .030 | .603 ± .023 | .955 ± .031 | .466 ± .038 | .739 ± .023 | .855 ± .025 | .891 ± .016 | .887 ± .041 | 70 | 5 |
| Always-positive dummy | .459 ± .007 | .459 ± .007 | 1.000 ± .000 | .000 ± .000 | .629 ± .007 | .809 ± .004 | .500 ± .000 | .459 ± .007 | 131 | 0 |

Outer confusion matrices in the order `[[TN, FP], [FN, TP]]` were `[[69, 62], [6, 105]]` for logistic regression, `[[61, 70], [5, 106]]` for random forest, and `[[0, 131], [0, 111]]` for the dummy.

Separate training-only configuration runs on all 242 rows chose:

- Logistic regression at threshold 0.30 with C 0.1, balanced class weights, and the fixed LBFGS solver.
- Random forest at threshold 0.30 with 200 trees, maximum depth 3, minimum leaf size 3, and balanced class weights.
- The dummy at its fixed threshold 0.50, with class 1 as the constant prediction.

The logistic and forest F2 means differ by about 0.005, smaller than their respective fold standard deviations. The forest missed one fewer positive at the cost of eight more false positives. Logistic specificity varied substantially across folds. The dummy gained perfect recall by flagging every negative, illustrating why recall alone is insufficient. The partial tuning and fold records are in [validation.json](artifacts/milestone-2-partial/validation.json).

## Verification and remaining work

- `python -m pytest -q -rs` produced **16 passed, 2 skipped, 10 subtests passed**. The skipped tests cover the XGBoost pipeline and rendering reports with Matplotlib.
- `python -m heart_disease prepare` produced a **242 × 28** training transformation. `pip check` and source compilation passed.
- A fitted logistic pipeline survived joblib serialization with identical inference on a training probe. Tests exercised feature order, null handling, and invalid inputs. The final selected artifact has not been generated.
- The dataset SHA-256 remains `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`. The split manifest SHA-256 remains `c903cdb3ab3db22380fde88ae72652ee1abc7addc1dab0d2280bbfba0024f2e3`. Fold checks covered train/validation separation and exclusion of reserved row IDs.
- No final model was selected and no holdout predictions, final evaluation figures, or final serialized model were generated. XGBoost and plotting require a follow-up run with both dependencies installed.

Use the online or offline PowerShell installation in [README.md](README.md#milestone-2-model-experiment), then run the full pytest command and experiment once from the project root.

## Closure based on the existing completed run

The original experiment directory contains the four trained candidate results, locked
configuration, fitted joblib artifact, holdout marker, saved predictions and figures.
The blocked partial run above remains historical. The completed run selected Logistic
Regression with C=0.1 and balanced class weights at threshold 0.30. Its saved confusion
matrix is [[25, 8], [1, 27]]. Verification recomputed metrics from the recorded scores,
checked selection against training-only validation, compared artifact metadata, and ran
both fictional inference examples. It did not execute the final holdout again.
