# Recruiter demo distribution

Release tag: [v0.1.0](https://github.com/azim-haffar/heart-disease-prediction-system/releases/tag/v0.1.0).

From the matching GitHub release in the project owner's repository, download its
source ZIP and manually attached `model.joblib` and `model.joblib.sha256` files.
The SHA-256 below is the reference in the reviewed source. Verify the owner's
repository and matching tag before downloading. Joblib files can execute code.
Only load this exact owner-provided model after verifying its checksum.

Extract the source ZIP, open PowerShell at its root, and use Windows Python 3.14.8.
The requirements file pins the observed Windows demo runtime. Runtime wheels and
packages must be available to pip. Alternate platforms and Python versions were
outside verification.

Before the checksum and launch commands, place the downloaded `model.joblib`
in `artifacts/milestone-2/` relative to the extracted source root. Create that
directory if needed. Use a fresh shell with DATABASE_URL unset for the demo.

```powershell
python --version
python -m venv .venv
$python = '.\.venv\Scripts\python.exe'
& $python -m pip install -r requirements-demo-windows.txt
if ($LASTEXITCODE) { throw 'Dependency installation failed' }
& $python -m pip check
if ($LASTEXITCODE) { throw 'Dependency consistency check failed' }
$env:PYTHONPATH = (Resolve-Path .\src).Path
$env:HEART_RUN_DIR = (Resolve-Path .\artifacts\milestone-2).Path
$expected = '404b8318f6b1af292cae21791f87171ef5c0ddd4c21689a390231b1eceba52e1'
$actual = (Get-FileHash (Join-Path $env:HEART_RUN_DIR 'model.joblib') -Algorithm SHA256).Hash
if ($actual -ne $expected) { throw 'Model checksum mismatch' }
& $python -m streamlit run app.py
```

Select **Prediction Demo** and run both fictional examples. A displays a score
of 0.068 and a negative class. B displays 0.919 and a positive class. The
decision threshold is 0.30. These are educational outputs for a dataset-defined
target. The minimal asset powers prediction and the methodology section. The
Model Comparison page requires the original private reports and shows an
unavailable notice in this distribution. Aggregate results are in RELEASE.md.
No training, holdout scoring, or database is needed to use this demo.

The model includes coefficients, preprocessing statistics and category codes,
feature metadata, and source-row positions already present in the frozen split.
The release assets exclude selection and validation reports, individual holdout
scores, charts, and database material. The review inspected the exact local
artifact. It provides no general guarantee about other joblib files.

The original software and its documentation are licensed under [MIT](LICENSE).
The Cleveland dataset remains under CC BY 4.0, with attribution in [README](README.md).
The release assets are distributed separately from the source tree.
