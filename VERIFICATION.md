# Milestone 1 verification

**Outcome: HOLD**

## Executed checks

| Check | Observed result |
| --- | --- |
| Dependency configuration | Inspected `pyproject.toml`. Runtime dependencies declare NumPy, pandas, and scikit-learn. The test extra declares pytest. Build isolation requires setuptools 77 or later. |
| Editable installation | Failed while resolving `pypi.org` for the isolated setuptools installation. Independent shell requests to PyPI and its wheel host `files.pythonhosted.org` also failed DNS resolution. |
| Dataset audit | Passed on Python 3.14.8. The snapshot checksum validated before parsing. All 303 rows were audited. |
| Frozen split reproduction | Passed. The regenerated manifest matched `data/split.json` byte for byte. SHA-256 `c903cdb3ab3db22380fde88ae72652ee1abc7addc1dab0d2280bbfba0024f2e3`. Training has 242 rows (131 absence, 111 presence). Final holdout has 61 rows (33 absence, 28 presence). |
| Dataset tests | Seven tests passed via `unittest`. |
| Training-only fit | Reviewed the source path. `prepare` passes only the training IDs into `training_features`, then fits on the resulting frame. It uses the holdout IDs to report class counts. Runtime fit and learned statistics could not be checked. |
| Preprocessing execution | Attempted on the bundled Python 3.12.14 runtime. Import failed with `ModuleNotFoundError: No module named 'sklearn'` before fitting. Numerical and categorical imputation, scaling, and unknown-category behavior remain unverified at runtime. |
| Complete pytest suite | Attempted on Python 3.12.14. The interpreter reported `No module named pytest` before collection. |
| Dataset provenance | All 303 ordered record lines matched the official [UCI text listing](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data) when compared as text lines with terminators omitted. Direct download for a raw-byte comparison failed DNS resolution in the local execution shell. |

The project virtual environment uses Python 3.14.8 with pip 26.2.1 and lacked the project dependencies. The bundled Python 3.12.14 runtime has NumPy 2.3.5, pandas 3.0.1, and setuptools 84.2.1. It lacks scikit-learn and pytest. The verification shell’s DNS failure occurred before package compatibility or project execution could be evaluated. DNS availability in the user’s regular PowerShell session was outside this check.

This verification changed documentation only. The dataset, frozen manifest, dependency configuration, application code, and test expectations were preserved.

## Complete verification in Windows PowerShell

From the local project directory, use a regular Windows PowerShell session with network access.

```powershell
# Run from the repository root.
python -m venv .venv
$python = '.\.venv\Scripts\python.exe'
& $python -m pip install -e '.[test]'
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
```

If the target machine is offline, download wheels on an internet-connected Windows machine with the same Python major/minor version and architecture. Copy the entire wheelhouse to the offline machine.

```powershell
python -m pip download --only-binary=:all: --dest .\wheelhouse 'setuptools>=77' wheel 'numpy>=1.26,<3' 'pandas>=2.2,<4' 'scikit-learn>=1.5,<2' 'pytest>=8,<10'
```

With the copied wheelhouse at `C:\wheelhouse`, run on the offline machine.

```powershell
# Run from the repository root.
python -m venv .venv
$python = '.\.venv\Scripts\python.exe'
& $python -m pip install --no-index --find-links 'C:\wheelhouse' -e '.[test]'
if ($LASTEXITCODE -ne 0) { throw 'Offline dependency installation failed' }
```

After either installation succeeds, run the same verification commands.

```powershell
& $python --version
& $python -m pip list --format=freeze
& $python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed' }
& $python -m heart_disease audit --output artifacts/verification-audit.json
if ($LASTEXITCODE -ne 0) { throw 'Audit failed' }
& $python -m heart_disease split --output artifacts/verification-split.json
if ($LASTEXITCODE -ne 0) { throw 'Split failed' }
if ((Get-FileHash data/split.json).Hash -ne (Get-FileHash artifacts/verification-split.json).Hash) {
    throw 'Frozen split differs from the regenerated split'
}
& $python -m heart_disease prepare --output artifacts/verification-preparation.json
if ($LASTEXITCODE -ne 0) { throw 'Preprocessing failed' }
(Get-Content artifacts/verification-preparation.json -Raw | ConvertFrom-Json).transformed_training_shape
& $python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Pytest failed' }
```

For a raw-byte source comparison when UCI is reachable, download to the ignored artifacts directory and compare hashes.

```powershell
New-Item -ItemType Directory -Force artifacts | Out-Null
curl.exe --fail --location --silent --show-error 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data' --output artifacts/official-cleveland.data
if ($LASTEXITCODE -ne 0) { throw 'Official source download failed' }
Get-FileHash data/processed.cleveland.data, artifacts/official-cleveland.data -Algorithm SHA256
```

## Later verification on the project environment

Milestone 1 was accepted after the project virtual environment gained its dependencies.
Milestone 2 reran the original pytest checks with Python 3.14.8, scikit-learn 1.9.1, and
pytest 9.2.0. The historical HOLD above describes the earlier execution environment.
The source provenance limitation concerning a raw-byte UCI download still applies.
