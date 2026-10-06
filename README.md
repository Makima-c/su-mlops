# su-mlops

A small API for predicting diabetes progression. You only need Docker;
all models are included in the images.

## Run with Docker

v0.1 — StandardScaler + LinearRegression:

```bash
docker pull ghcr.io/makima-c/su-mlops:v0.1
docker run --rm -p 80:80 ghcr.io/makima-c/su-mlops:v0.1
```

Press Ctrl+C to stop v0.1 before running v0.2, since both use port 80.

v0.2 — RandomForestRegressor:

```bash
docker pull ghcr.io/makima-c/su-mlops:v0.2
docker run --rm -p 80:80 ghcr.io/makima-c/su-mlops:v0.2
```

To build and run v0.2 from this repository instead:

```bash
docker compose up --build -d
```

Compose uses port 3000 from `.env`. Use `localhost:3000` for the examples below.
Stop it with `docker compose down`.

## Try the API

Run these in another terminal. Check the running version:

```bash
curl http://localhost:80/health
```

Response: `{"status":"ok","model_version":"v0.2"}` (or `v0.1`, depending on the image).

Send all 10 features to get a prediction:

```bash
curl -X POST http://localhost:80/predict \
  -H "Content-Type: application/json" \
  -d '{"age":0.02,"sex":-0.044,"bmi":0.06,"bp":-0.03,"s1":-0.02,"s2":0.03,"s3":-0.02,"s4":0.02,"s5":0.02,"s6":-0.001}'
```

Example v0.2 response: `{"prediction":178.1797683288933}`.
Higher values mean worse predicted progression.

Missing or incorrect features return HTTP **422**. For example, missing `s6` returns:

```json
{"error":"Invalid input","details":{"s6":"Required field is missing"}}
```

`details` lists the invalid fields. Invalid JSON or a non-object body uses `details.body`.

See [CHANGELOG.md](CHANGELOG.md) for model changes, RMSE comparisons and training commands.
