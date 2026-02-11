# Kubernetes Quickstart

This folder deploys the phase-2 stack to Kubernetes:
- Redpanda (Kafka)
- Producer
- Aggregator
- Failure-risk trainer (Job)
- Failure-risk service

## 1) Build images locally

```bash
cd <repo-root>

docker compose build producer aggregator

docker build -t predictive_maintaince_platform-failure-risk-trainer:latest app/failure_risk_trainer
docker build -t predictive_maintaince_platform-failure-risk-service:latest app/failure_risk_service
```

## 2) Load images into Kind

```bash
kind load docker-image predictive_maintaince_platform-producer:latest
kind load docker-image predictive_maintaince_platform-aggregator:latest
kind load docker-image predictive_maintaince_platform-failure-risk-trainer:latest
kind load docker-image predictive_maintaince_platform-failure-risk-service:latest
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

## 4) Train failure-risk model

```bash
kubectl apply -f k8s/07-failure-risk-trainer-job.yaml
kubectl -n predictive-maintenance logs job/failure-risk-trainer -f
```

## 5) Start failure-risk service

```bash
kubectl apply -f k8s/08-failure-risk-service.yaml
kubectl -n predictive-maintenance port-forward svc/failure-risk-service 8000:8000
```

In another terminal:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model/info
curl http://localhost:8000/predict/latest/M-001
```
