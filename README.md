# Log Anomaly Detector: Log Anomaly Detection via BERT
### [ARXIV](https://arxiv.org/abs/2103.04475) 
<!-- Description -->
A FastAPI service wrapping a LogBERT-based pipeline for detecting anomalies in raw log streams. Includes preprocessing, sequence modeling, and an HTTP API for health and anomaly detection.

This repository provides the implementation of Logbert for log anomaly detection. 
The process includes downloading raw data online, parsing logs into structured data, 
creating log sequences and finally modeling. 

![alt](img/log_preprocess.png)

## Configuration
- Ubuntu 20.04
- NVIDIA driver 460.73.01 
- CUDA 11.2
- Python 3.8
- PyTorch 1.9.0

## Installation
This code requires the packages listed in requirements.txt.
An virtual environment is recommended to run this code

On macOS and Linux:  
```
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate
pip install -r ./environment/requirements.txt
deactivate
```
Reference: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/

An alternative is to create a conda environment:
```
    conda create -f ./environment/environment.yml
    conda activate logbert
```
Reference: https://docs.conda.io/en/latest/miniconda.html

## Experiment
Logbert and other baseline models are implemented on [HDFS](https://github.com/logpai/loghub/tree/master/HDFS), [BGL](https://github.com/logpai/loghub/tree/master/BGL), and [thunderbird]() datasets

### HDFS example
```shell script

cd HDFS

sh init.sh

# process data
python data_process.py

#run logbert
python logbert.py vocab
python logbert.py train
python logbert.py predict

#run deeplog
python deeplog.py vocab
# set options["vocab_size"] = <vocab output> above
python deeplog.py train
python deeplog.py predict 

#run loganomaly
python loganomaly.py vocab
# set options["vocab_size"] = <vocab output> above
python loganomaly.py train
python loganomaly.py predict

#run baselines

baselines.ipynb
```

### Folders created during execution
```shell script 
~/.dataset //Stores original datasets after downloading
project/output //Stores intermediate files and final results during execution
```

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
