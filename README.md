# su-mlops

MAIO Assignment 3: diabetes progression prediction using StandardScaler + LinearRegression.

## Run

Requires Docker with Compose. Run from the repository root:

```bash
docker compose up --build -d
```

The model is included in the image; no retraining is needed. Port is set by
`PORT=3000` in `.env`.

The standalone Docker image defaults to port 80:

```bash
docker build -t su-mlops:v0.1 .
docker run --rm -p 80:80 su-mlops:v0.1
```

For this command, use `localhost:80` in the API examples below.

Stop:

```bash
docker compose down
```

## API examples

Health check:

```bash
curl http://localhost:3000/health
```

Response: `{"status":"ok","model_version":"v0.1"}`.

Prediction:

```bash
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":0.02,"sex":-0.044,"bmi":0.06,"bp":-0.03,"s1":-0.02,"s2":0.03,"s3":-0.02,"s4":0.02,"s5":0.02,"s6":-0.001}'
```

Example response (rounded): `{"prediction":235.949637}`. Higher means worse progression.

Missing `s6`:

```bash
curl -i -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":0.02,"sex":-0.044,"bmi":0.06,"bp":-0.03,"s1":-0.02,"s2":0.03,"s3":-0.02,"s4":0.02,"s5":0.02}'
```

Returns HTTP **422**:

```json
{ "error": "Invalid input", "details": { "s6": "Required field is missing" } }
```

Other invalid inputs use the same `error` and `details` structure;
body-level errors use `details.body`.

## Train locally

Requires uv and Python 3.13.7:

```bash
uv sync --locked
uv run --locked python train-baseline.py
```

Exports `model.pkl`, `metrics.json` and `metadata.json` to `models/v0.1/`.
Training uses an 80/20 split with seed 42 and reports held-out RMSE.

## Test

```bash
uv run --locked python -m pytest -q
```

PRs and branch pushes run lint, training and server tests in GitHub Actions.
Successful runs save the model, metrics and metadata as `baseline-model` artifacts.

The same workflow runs CI for `v*` tags, then CD uses the saved model to build,
smoke-test and publish the version tag and `latest` to GHCR, plus a GitHub Release
with changelog and metrics. The tag must match the model version. After the first
publish, set the GHCR package visibility to Public.
