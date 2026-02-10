# Kubernetes Quickstart

This folder deploys the phase-2 stack to Kubernetes:
- Redpanda (Kafka)
- Producer
- Aggregator
- Trainer (Job)
- Model service

## 1) Build images locally

```bash
cd /Users/nelson/py/predictive_maintaince_platform

docker compose build producer aggregator

docker build -t predictive_maintaince_platform-trainer:latest app/failure_risk_trainer
docker build -t predictive_maintaince_platform-model-service:latest app/failure_risk_model_service
```

## 2) Load images into Kind

```bash
kind load docker-image predictive_maintaince_platform-producer:latest
kind load docker-image predictive_maintaince_platform-aggregator:latest
kind load docker-image predictive_maintaince_platform-trainer:latest
kind load docker-image predictive_maintaince_platform-model-service:latest
```

## 3) Deploy base services

```bash
kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/01-storage.yaml
kubectl apply -f k8s/02-redpanda.yaml
kubectl apply -f k8s/03-init-topic-job.yaml
kubectl apply -f k8s/04-configmap.yaml
kubectl apply -f k8s/05-producer.yaml
kubectl apply -f k8s/06-aggregator.yaml
```

## 4) Train model

```bash
kubectl apply -f k8s/07-trainer-job.yaml
kubectl -n predictive-maintenance logs job/trainer -f
```

## 5) Start model service

```bash
kubectl apply -f k8s/08-model-service.yaml
kubectl -n predictive-maintenance port-forward svc/model-service 8000:8000
```

In another terminal:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model/info
curl http://localhost:8000/predict/latest/M-001
```
