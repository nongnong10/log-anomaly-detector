# Log Anomaly Detector: Log Anomaly Detection via BERT
<!-- Description -->
## Docker
Description:
Builds a CPU-only image (python:3.10 + torch==2.2.1) and serves the API on port 8000.

Build:
```bash
docker build -t log-anomaly-detector:latest .
```
Run:
```bash
docker run -p 8000:8000 log-anomaly-detector:latest
```
Health check:
```bash
curl http://localhost:8000/health
```
Anomaly detect example:
```bash
curl -X POST http://localhost:8000/detect -H 'Content-Type: application/json' \
  -d '{"raw_log_data":"2024-11-01 00:00:00 123 INFO HDFS: blk_-1 started","seq_threshold":0.2}'
```
Previous libtorch_cpu.so executable stack error was resolved by:
- Using python:3.10 base.
- Installing torch==2.2.1 (CPU wheel) via PyTorch CPU index.
- Adding libgomp1 and libstdc++6.

Optional hardened runtime test:
```bash
docker run --security-opt seccomp=unconfined -p 8000:8000 log-anomaly-detector:latest
```

## Push Image to Google Cloud
Prerequisites: gcloud CLI installed and a Google Cloud project.

1. Authenticate and set project:
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```
2. Enable Artifact Registry:
```bash
gcloud services enable artifactregistry.googleapis.com
```
3. Create a Docker repository (once):
```bash
gcloud artifacts repositories create log-anomaly-detector \
  --repository-format=docker --location=us-central1 \
  --description="Log anomaly detector service"
```
4. Build locally (if not already):
```bash
docker build -t log-anomaly-detector:latest .
```
5. Tag for Artifact Registry:
```bash
docker tag log-anomaly-detector:latest \
  us-central1-docker.pkg.dev/YOUR_PROJECT_ID/log-anomaly-detector/log-anomaly-detector:latest
```
6. Push:
```bash
docker push \
  us-central1-docker.pkg.dev/YOUR_PROJECT_ID/log-anomaly-detector/log-anomaly-detector:latest
```
7. (Optional) Deploy to Cloud Run:
```bash
gcloud run deploy log-anomaly-detector \
  --image us-central1-docker.pkg.dev/YOUR_PROJECT_ID/log-anomaly-detector/log-anomaly-detector:latest \
  --platform managed --region us-central1 --allow-unauthenticated --port 8000
```
8. Get deployed URL:
```bash
gcloud run services describe log-anomaly-detector --region us-central1 --format 'value(status.url)'
```

### Live API (Cloud Run)
Deployed interactive docs (Swagger/OpenAPI):
https://log-anomaly-detector-183907812838.us-central1.run.app/docs

### Database Setup (PostgreSQL)
```bash
docker run -d \
        --name some-postgres \
        -e POSTGRES_PASSWORD=123456 \
        -e POSTGRES_USER=postgres \
        -e POSTGRES_DB=log_anomaly_detector \
        -p 5432:5432 \
        postgres
```