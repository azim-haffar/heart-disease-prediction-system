"""Display labels and fictional examples for the Cleveland inference schema."""

from .dataset import FEATURES

NUMBERS = {
    "age": ("Age (years)", "Age at the time of the historical observation."),
    "trestbps": ("Resting blood pressure (mm Hg)", "Recorded on hospital admission in the source data."),
    "chol": ("Serum cholesterol (mg/dL)", "Laboratory serum cholesterol measurement."),
    "thalach": ("Peak heart rate (beats/min)", "Maximum heart rate achieved during exercise testing."),
    "oldpeak": ("Exercise ST depression", "Exercise-induced ST depression relative to rest."),
}
OPTIONS = {
    "sex": ("Recorded sex", {0: "Female", 1: "Male"}),
    "cp": ("Chest pain type", {1: "Typical angina", 2: "Atypical angina",
                             3: "Non-anginal pain", 4: "Asymptomatic"}),
    "fbs": ("Fasting blood sugar above 120 mg/dL", {0: "No", 1: "Yes"}),
    "restecg": ("Resting ECG", {0: "Normal", 1: "ST–T wave abnormality",
                             2: "Probable or definite left ventricular hypertrophy"}),
    "exang": ("Exercise-induced angina", {0: "No", 1: "Yes"}),
    "slope": ("Peak exercise ST segment slope", {1: "Upsloping", 2: "Flat", 3: "Downsloping"}),
    "ca": ("Major vessels visible on fluoroscopy", {0: "0", 1: "1", 2: "2", 3: "3", None: "Unknown"}),
    "thal": ("Thallium stress test", {3: "Normal", 6: "Fixed defect",
                                    7: "Reversible defect", None: "Unknown"}),
}

PROFILES = {
    "Synthetic example A": dict(zip(FEATURES, (49, 0, 2, 117, 203, 0, 0, 167, 0, 0.4, 1, 0, 3))),
    "Synthetic example B": dict(zip(FEATURES, (62, 1, 4, 151, 260, 0, 2, 128, 1, 2.2, 2, None, 7))),
}

MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "always_positive": "Always-positive baseline",
}
