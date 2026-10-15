# Deployment

The app is hosted on Vercel as two projects.

| Part | Project | URL |
|---|---|---|
| Frontend (React + Vite) | `loan-default-risk` | <https://delinquency-risk.vercel.app> |
| Backend (FastAPI + model) | `loan-default-api` | <https://loan-default-api.vercel.app> (Swagger at `/docs`) |

## How the two talk to each other

The browser never calls the backend directly. The frontend's `vercel.json`
rewrites `/api/*` to the backend, so every request is **same-origin** and CORS
does not apply:

```
browser ──► delinquency-risk.vercel.app/api/predict
                    │  (Vercel rewrite, server side)
                    ▼
            loan-default-api.vercel.app/predict
```

Production builds read `VITE_API_BASE_URL=/api` from `frontend/.env.production`.
Local development is unchanged and still calls `http://127.0.0.1:8000`.

CORS is also configured on the backend as a second layer, through the
`CORS_ORIGINS` environment variable (comma-separated). It currently allows the
frontend's two production domains and the local dev server; any other origin is
rejected.

## Why the backend fits on Vercel

Vercel's Python bundle limit is 500 MB unzipped. The deployed bundle is about
315 MB: the installed dependencies (numpy, pandas, scipy, scikit-learn,
FastAPI and pydantic) are roughly 245 MB and the 70 MB model file is the rest.

That only works because `backend/requirements.txt` holds just what serving
needs. Training, plotting and test tools live in `requirements-dev.txt`, which
the deployment never installs. `backend/.vercelignore` additionally keeps
training scripts, tests and figures out of the bundle. `model.joblib` is
git-ignored but is deliberately **not** in `.vercelignore`, so it ships.

## Things worth knowing

- **Cold start.** After the backend has been idle, the first request takes
  about 8 to 9 seconds while the 70 MB model loads. Later requests take
  roughly half a second. The frontend shows its loading spinner meanwhile.
- **The model is baked into each deployment.** Retraining locally changes
  nothing online until the backend is redeployed.
- **Deployment-specific URLs are private.** URLs such as
  `loan-default-api-<hash>-....vercel.app` sit behind Vercel login. The
  production aliases above are public.
- **No authentication.** The API is open. That suits a coursework demo, not
  anything handling real applicants.

## Redeploying

From the project root, with the Vercel CLI logged in:

```bash
cd backend  && vercel deploy --prod --yes     # API
cd frontend && vercel deploy --prod --yes     # website
```

Each directory is already linked to its project through `.vercel/`, which is
git-ignored, so on a fresh clone run `vercel link --project <name>` first.

## A gotcha when setting environment variables on Windows

Do **not** pipe a value into `vercel env add` from Windows PowerShell 5.1:

```powershell
"/api" | vercel env add VITE_API_BASE_URL production     # adds a hidden BOM
```

PowerShell prepends a UTF-8 byte-order mark, so the stored value becomes
`\uFEFF/api` and every request URL starts with `%EF%BB%BF`. The site then
fetches `/%EF%BB%BF/api/model-info`, receives the app's own HTML page, and
fails with `Unexpected token '<'`. This actually happened here. Use the flag
instead, and prefer committing non-secret values in `.env.production`:

```powershell
vercel env add CORS_ORIGINS production --value "https://example.vercel.app" --yes
```

`frontend/src/services/api.ts` also trims the configured base URL as a guard.

## Verifying a deployment

A `curl` against `/api/...` proves the proxy works but **not** that the built
site calls the right URLs, which is how the BOM bug slipped past the first
check. Load the site in a browser and confirm that the network tab shows
`GET /api/model-info` returning 200 and that the console is empty.

## Responsive layout

The site was checked at 320px (small phone), 375px (typical phone), 768px
(tablet) and 1280px (desktop). At each width the page must not scroll
sideways, and the charts must render at 1:1 scale. Two things caused real
problems on phones and are worth remembering:

- **Unbroken strings force the page wider.** The long Kaggle URL made the
  Dashboard and Model info pages scroll sideways at 375px. Long values need
  `break-all` or `break-words`, and flex or grid children that hold them need
  `min-w-0`.
- **SVG text scales with the drawing.** A chart given a fixed `viewBox` shrinks
  its labels along with the lines. Measure the container and draw at that width.

The header navigation wraps onto a second line on very narrow screens instead
of hiding a section behind a sideways scroll.
