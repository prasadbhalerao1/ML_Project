# Setup and run guide

Step-by-step instructions to get this project running from a fresh clone.
Follow the five steps in order; the whole thing takes about 10 minutes, most of
it model training.

For what the project *is* and what the results mean, see
[README.md](../README.md).

## Prerequisites

| Requirement | Version used here | Check with |
|---|---|---|
| Python | 3.14.3 (3.11+ works) | `python --version` |
| Node.js | 22.17.1 (18+ works) | `node --version` |
| npm | 11.4.2 | `npm --version` |
| A Kaggle account | — | needed for step 1 |

On Windows, if `python` opens the Microsoft Store instead of running, use
`py` in place of `python` throughout, or install Python from
[python.org](https://www.python.org/downloads/) with "Add to PATH" ticked.

---

## Step 1 — Download the dataset

The raw CSVs are **not** included in the repository. You must download them
from Kaggle yourself.

### Option A — browser (simplest)

1. Sign in at <https://www.kaggle.com>.
2. Open <https://www.kaggle.com/competitions/GiveMeSomeCredit/data>.
3. Go to the **Rules** tab and click **I Understand and Accept**.
   Downloads are blocked until you do this.
4. Download **`cs-training.csv`** (the only file needed for modelling).
5. Put it in `data/raw/GiveMeSomeCredit/`, creating the folders if needed.

### Option B — Kaggle API

```bash
pip install kaggle
```

Get a token: Kaggle → your avatar → **Settings** → **API** → **Create New
Token**. This downloads `kaggle.json`. Save it to:

- **Windows:** `C:\Users\<you>\.kaggle\kaggle.json`
- **macOS / Linux:** `~/.kaggle/kaggle.json`, then `chmod 600 ~/.kaggle/kaggle.json`

Then, from the project root:

```bash
kaggle competitions download -c GiveMeSomeCredit -p data/raw/GiveMeSomeCredit
cd data/raw/GiveMeSomeCredit
unzip GiveMeSomeCredit.zip
cd ../../..
```

> The API returns **403 Forbidden** until you have accepted the competition
> rules in a browser at least once. Do step A3 first.

### Verify

You should have:

```
data/raw/GiveMeSomeCredit/
├── cs-training.csv        ← required (150,000 labeled rows, ~7.5 MB)
├── cs-test.csv            ← optional, unlabeled, not used
├── Data Dictionary.xls    ← optional
└── sampleEntry.csv        ← optional
```

---

## Step 2 — Create the virtual environment and install

Run these from the **project root**.

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r backend/requirements-dev.txt
```

If PowerShell blocks the activation script with an execution-policy error:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r backend/requirements-dev.txt
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements-dev.txt
```

Your prompt should now start with `(.venv)`. Installation takes 1–2 minutes
and pulls in pandas, scikit-learn, FastAPI, matplotlib and pytest.
`requirements-dev.txt` includes the lean runtime file `requirements.txt` (what the
hosted API installs) plus training, plotting and test tools.

> **Don't run `pip install --upgrade pip` inside a OneDrive-synced folder.**
> OneDrive locks files mid-write and the upgrade can corrupt pip, leaving you
> with `ModuleNotFoundError: No module named 'pip._internal.cli'`. If that
> happens, delete `.venv` and recreate it without upgrading pip. The pinned
> requirements install fine on the bundled pip.

---

## Step 3 — Train the model

**All backend commands must be run from the `backend/` directory** — the code
imports `ml.*` and `app.*` as packages, which only resolves from there.

```bash
cd backend
python -m ml.train
```

Takes about a minute. It fits five Random Forest configurations, picks one on
the validation split, selects the decision threshold, scores the held-out test
split once, and writes to `backend/ml/artifacts/`:

| File | What it is |
|---|---|
| `model.joblib` | The fitted pipeline (~75 MB, git-ignored) |
| `metadata.json` | Feature names, threshold, metrics, importances |
| `demo_examples.json` | Two real records for the UI's demo button |

You should see output ending with:

```
Held-out test metrics (threshold 0.2302)
  accuracy   0.9199
  precision  0.4153
  recall     0.4868
  f1         0.4482
  roc_auc    0.8654
  pr_auc     0.4014
```

### Optional extras

```bash
python -m ml.explore                    # data inspection + EDA charts
python -m ml.evaluate                   # per-split metrics, classification report
python -m ml.evaluate --cross-validate  # the above plus 5-fold CV (slower)
```

---

## Step 4 — Run the backend API

From the `backend/` directory, with the venv active:

```bash
uvicorn app.main:app --reload --port 8000
```

Leave this terminal running. Check it works:

- API docs (Swagger): <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/health>

`/health` should return:

```json
{"status":"ok","model_loaded":true,"model_version":"1.0.0","detail":null}
```

If `model_loaded` is `false`, step 3 didn't produce an artifact — the `detail`
field will say why.

---

## Step 5 — Run the frontend

Open a **second terminal** (leave the API running in the first). The venv is
not needed here.

```bash
cd frontend
npm install
npm run dev
```

`npm install` takes a minute the first time. Then open
**<http://localhost:5173>**.

You should see the Dashboard with live metrics loaded from the API. Click
**Predict** → **higher-risk example** → **Predict risk** to check the full
round trip.

### API URL configuration (optional)

The frontend defaults to `http://127.0.0.1:8000`, so **no configuration is
needed** if you followed step 4. To point it elsewhere:

```bash
cp .env.example .env     # Windows: copy .env.example .env
```

Then edit `.env`:

```
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Restart `npm run dev` after changing it — Vite reads env files at startup.

---

## Running the tests

With the venv active, from `backend/`:

```bash
pytest
```

Expect **61 passed**. Tests that need a trained model skip cleanly if you
haven't run step 3 yet.

Frontend checks, from `frontend/`:

```bash
npm run typecheck     # TypeScript, no emit
npm run build         # production build
```

---

## Command reference

| Task | Directory | Command |
|---|---|---|
| Create venv | root | `python -m venv .venv` |
| Activate venv | root | `.venv\Scripts\Activate.ps1` (Win) / `source .venv/bin/activate` |
| Install deps | root | `pip install -r backend/requirements-dev.txt` |
| Inspect data | `backend` | `python -m ml.explore` |
| Train model | `backend` | `python -m ml.train` |
| Evaluate | `backend` | `python -m ml.evaluate --cross-validate` |
| Run tests | `backend` | `pytest` |
| Start API | `backend` | `uvicorn app.main:app --reload --port 8000` |
| Install frontend | `frontend` | `npm install` |
| Start frontend | `frontend` | `npm run dev` |
| Build frontend | `frontend` | `npm run build` |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'ml'` or `'app'`**
You're in the wrong directory. All Python commands run from `backend/`, not the
project root.

**`DatasetError: Training data not found at ...`**
Step 1 is incomplete. Check that `cs-training.csv` is at
`data/raw/GiveMeSomeCredit/cs-training.csv` — note the exact folder name and
that the file is the *training* one.

**`ArtifactError: No model artifact in ...`**
Run `python -m ml.train` from `backend/` first.

**Frontend shows "Could not reach the API at http://127.0.0.1:8000"**
The backend isn't running, or it's on a different port. Check the first
terminal, and confirm <http://127.0.0.1:8000/health> loads in a browser.

**Frontend loads but all metrics are missing**
The API is up but has no model. Check `/health` — if `model_loaded` is false,
re-run step 3.

**`403 Forbidden` from the Kaggle CLI**
Accept the competition rules in a browser first (step 1, option A, item 3).

**Port 8000 or 5173 already in use**
Use a different one: `uvicorn app.main:app --port 8001`, and set
`VITE_API_BASE_URL=http://127.0.0.1:8001` in `frontend/.env`. For the
frontend, `npm run dev -- --port 5174`.

**PowerShell: "running scripts is disabled on this system"**
See the `Set-ExecutionPolicy` note in step 2.
