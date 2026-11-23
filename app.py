"""Streamlit demonstration backed by a trusted, local experiment."""

import os
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from heart_disease.dataset import FEATURES
from heart_disease.demo import MODEL_NAMES, NUMBERS, OPTIONS, PROFILES
from heart_disease.inference import load_trusted_model
from heart_disease.prevention import prevention_visit
from heart_disease.saved_run import read_saved_run

ROOT = Path(__file__).resolve().parent
RUN = Path(os.environ.get("HEART_RUN_DIR", ROOT / "artifacts" / "milestone-2"))
NOTICE = "Educational demonstration using historical data. Outputs provide neither a clinical diagnosis nor a validated estimate of future disease risk."


@st.cache_resource
def model_for(path, modified_ns, length):
    return load_trusted_model(path)


def prediction():
    st.header("Prediction demo")
    st.caption(NOTICE)
    st.write("Use a fictional example to explore the 13 predictors and the model’s decision threshold.")
    profile = st.selectbox("Fictional example", list(PROFILES))
    values = {}
    with st.form(f"prediction-{profile}"):
        columns = st.columns(2)
        for index, name in enumerate(FEATURES):
            default = PROFILES[profile][name]
            with columns[index % 2]:
                if name in NUMBERS:
                    label, help_text = NUMBERS[name]
                    values[name] = st.number_input(
                        label, min_value=0.0 if name == "oldpeak" else 1,
                        value=default, step=0.1 if name == "oldpeak" else 1,
                        help=help_text, key=f"{profile}-{name}",
                    )
                else:
                    label, choices = OPTIONS[name]
                    values[name] = st.selectbox(
                        label, list(choices), index=list(choices).index(default),
                        format_func=choices.__getitem__, key=f"{profile}-{name}",
                    )
        submitted = st.form_submit_button("Run prediction", type="primary")
    path = RUN / "model.joblib"
    if not path.is_file():
        st.warning("The saved model is missing. Restore the verified experiment directory to enable predictions.")
        return
    if submitted:
        try:
            info = path.stat()
            model = model_for(path, info.st_mtime_ns, info.st_size)
            result = model.predict(values)
        except (ValueError, TypeError, OSError, ImportError, AttributeError, EOFError, KeyError, pickle.UnpicklingError):
            st.error("Prediction could not run. Check the form values and restore the compatible, trusted model artifact if needed.")
            return
        label = "Positive" if result["classification"] else "Negative"
        st.subheader(f"{label} for the dataset-defined target")
        name = MODEL_NAMES.get(model.metadata["algorithm"], model.metadata["algorithm"])
        st.write(f"{name} · v{model.metadata['model_version']}")
        st.metric("Model score", f"{result['score']:.3f}")
        st.caption(f"Decision threshold {result['threshold']:.2f}. The score describes model output.")


def catalog_results():
    url = os.getenv("DATABASE_URL")
    if not url:
        return {}
    try:
        from sqlalchemy.exc import SQLAlchemyError
        from heart_disease.tracking import engine_from_url, fingerprint, read_results
    except ImportError:
        st.caption("Showing local reports. Optional database dependencies are unavailable.")
        return {}
    engine = None
    try:
        engine = engine_from_url(url)
        return read_results(engine, fingerprint(RUN))
    except (SQLAlchemyError, OSError, ValueError):
        st.caption("Showing local reports. The experiment database is unavailable.")
        return {}
    finally:
        if engine is not None:
            engine.dispose()


def comparison():
    st.header("Model comparison")
    try:
        selection, validation, heldout = read_saved_run(RUN)
    except (OSError, ValueError):
        st.warning("Saved evaluation reports are unavailable. Restore the verified experiment directory.")
        return
    catalog = catalog_results()
    expected = {("validation", model): result["summary"]
                for model, result in validation["outer_comparison"].items()}
    expected["holdout", selection["algorithm"]] = heldout["metrics"]
    if catalog == expected:
        st.caption("Metrics retrieved from the registered experiment catalog.")
    elif catalog:
        st.caption("Showing local reports. The experiment catalog disagrees with the saved run.")
        catalog = {}
    name = MODEL_NAMES[selection["algorithm"]]
    st.subheader("Training-only validation")
    st.write(f"{name} was selected using mean outer-fold F2. Candidates within 0.01 F2 "
             "of the best result were eligible for the predefined simplicity preference.")
    table = []
    for model, value in validation["outer_comparison"].items():
        row = {"Model": MODEL_NAMES[model]}
        for metric, label in (("accuracy", "Accuracy"), ("recall", "Recall"),
                              ("precision", "Precision"), ("specificity", "Specificity"),
                              ("f1", "F1"), ("f2", "F2"), ("roc_auc", "ROC-AUC"), ("average_precision", "AP")):
            stats = catalog.get(("validation", model), value["summary"])[metric]
            row[label] = f"{stats['mean']:.3f} ± {stats['sd']:.3f}"
        table.append(row)
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    st.caption("F2 emphasizes recall while accounting for precision. Values are fold means ± sample standard deviations.")
    if (RUN / "thresholds.png").is_file():
        st.image(str(RUN / "thresholds.png"), caption="Recall and precision over training-only thresholds")
    st.subheader("Recorded final holdout")
    st.write(f"{name} at threshold {selection['threshold']:.2f} · {len(heldout['row_ids'])} rows")
    result = catalog.get(("holdout", selection["algorithm"]), heldout["metrics"])
    cols = st.columns(3)
    for col, label, metric in zip(cols, ("Accuracy", "Recall", "Precision"), ("accuracy", "recall", "precision")):
        col.metric(label, f"{result[metric]:.1%}")
    st.write(f"F2 **{result['f2']:.3f}** · Specificity **{result['specificity']:.1%}**")
    st.write(f"F1 **{result['f1']:.3f}** | ROC-AUC **{result['roc_auc']:.3f}** | "
             f"Average precision **{result['average_precision']:.3f}**")
    st.write(f"False positives **{result['fp']}** (negative target classified positive) · "
             f"False negatives **{result['fn']}** (positive target classified negative)")
    for filename, caption in (("confusion-matrix.png", "Holdout confusion matrix"),
                              ("holdout-curves.png", "Holdout ROC, precision–recall and descriptive calibration")):
        if (RUN / filename).is_file():
            st.image(str(RUN / filename), caption=caption)
    st.caption(NOTICE)


def methodology():
    st.header("Dataset and methodology")
    st.caption(NOTICE)
    st.markdown("""
The UCI Cleveland snapshot contains **303 observations and 13 predictors**. The source
target maps code 0 to absence and codes 1–4 to presence. The frozen stratified split
reserves 61 observations for the recorded final evaluation.

- **Preparation.** Median imputation and scaling for numeric features. Most-frequent
  imputation and one-hot encoding for categorical features. Preprocessing fits separately
  in each training fold.
- **Comparison.** Five outer folds and three inner folds compare Logistic Regression,
  Random Forest, XGBoost, and an always-positive baseline using seed 42.
- **Threshold.** Inner out-of-fold scores guide threshold selection using F2.
  The final threshold was locked before the holdout was scored.
- **Limits.** The small historical cohort and finite holdout introduce substantial
  uncertainty. External validation and clinical calibration are unestablished.

[UCI Heart Disease dataset and attribution](https://doi.org/10.24432/C52P4X)
""")


st.set_page_config(page_title="Cleveland Heart Disease Explorer", layout="wide")
st.title("Cleveland Heart Disease Explorer")
page = st.radio("Section", ["Prediction Demo", "Model Comparison", "Dataset and Methodology", "Prevention Visit"],
                horizontal=True)
if page == "Prediction Demo":
    prediction()
elif page == "Model Comparison":
    comparison()
elif page == "Prevention Visit":
    prevention_visit()
else:
    methodology()
