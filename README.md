# 🔍 Fraud Detection MLOps Pipeline

An end-to-end, production-grade **MLOps pipeline** for real-time fraud detection,
combining machine learning with modern DevOps practices — experiment tracking,
containerized microservices, Kubernetes orchestration, and live monitoring.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange?logo=mlflow)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Container-blue?logo=docker)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Orchestration-blue?logo=kubernetes)
![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-orange?logo=prometheus)
![Grafana](https://img.shields.io/badge/Grafana-Dashboards-yellow?logo=grafana)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [ML Pipeline Design](#3-ml-pipeline-design)
4. [Tech Stack](#4-tech-stack)
5. [Directory Structure](#5-directory-structure)
6. [Setup & Installation](#6-setup--installation)
7. [Running Services](#7-running-services)
8. [MLflow Experiment Tracking](#8-mlflow-experiment-tracking)
9. [Docker & Kubernetes Deployment](#9-docker--kubernetes-deployment)
10. [Monitoring Stack](#10-monitoring-stack)
11. [Model Details & Results](#11-model-details--results)
12. [Future Roadmap](#12-future-roadmap)
13. [Contact](#13-contact)

---

## 1. Project Overview

### Problem Statement
Financial fraud causes billions in losses annually. Traditional rule-based systems
fail to catch sophisticated fraud patterns. This project builds an **intelligent,
adaptive fraud detection system** that learns from transaction patterns and
predicts fraud in real-time.

### What Makes This Production-Grade
- **Reproducibility** — every experiment is logged, versioned, and comparable via MLflow
- **Scalability** — Kubernetes orchestrates auto-scaling microservices
- **Observability** — Prometheus + Grafana give live visibility into model health
- **Modularity** — each component (data, model, API, UI, monitoring) is fully decoupled
- **CI/CD Ready** — Docker images deployable to any cloud (AWS/GCP/Azure)

### Key Results
- **98% accuracy** on balanced hold-out set
- **100% recall** on fraud cases (zero missed frauds in testing)
- **Optimal threshold: 0.8370** — precision 95.5%, recall 99.1%
- Real-time predictions served at sub-100ms latency via FastAPI

---

## 2. System Architecture

### High-Level Architecture
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                             │
│         Streamlit UI (port 8501)   │   External API Consumers   │
└─────────────────┬───────────────────────────────┬───────────────┘
│                               │
┌─────────────────▼───────────────────────────────▼───────────────┐
│                      SERVICE LAYER                              │
│              FastAPI REST API (port 8000)                       │
│         /predict  │  /health  │  /metrics                       │
└─────────────────┬───────────────────────────────────────────────┘
│
┌─────────────────▼───────────────────────────────────────────────┐
│                       MODEL LAYER                               │
│   FraudPipeline: Feature Eng → Preprocess → SMOTE → Model      │
│              MLflow Model Registry & Artifacts                  │
└─────────────────┬───────────────────────────────────────────────┘
│
┌─────────────────▼───────────────────────────────────────────────┐
│                  INFRASTRUCTURE LAYER                           │
│    Docker Containers → Kubernetes Pods (Minikube)               │
│    Prometheus (9090) → Grafana Dashboards (3000)                │
└─────────────────────────────────────────────────────────────────┘

### Microservices Breakdown

| Service | Port | Technology | Responsibility |
|---------|------|-----------|----------------|
| Streamlit UI | 8501 | Python/Streamlit | Interactive prediction interface |
| FastAPI Server | 8000 | Python/FastAPI | REST API for predictions |
| MLflow Server | 5000 | MLflow | Experiment tracking & model registry |
| Prometheus | 9090 | Prometheus | Metrics scraping & alerting |
| Grafana | 3000 | Grafana | Metrics visualization |

---

## 3. ML Pipeline Design

### End-to-End Flow
Raw Transaction Data
│
▼
┌───────────────────┐
│ Feature Engineering│  ← Interaction features, ratio features,
│                   │    time-of-day binning, account age buckets
└────────┬──────────┘
│
▼
┌───────────────────┐
│   Preprocessing   │  ← Imputation (median/mode), One-hot encoding,
│                   │    Log transform, StandardScaler + MinMaxScaler
└────────┬──────────┘
│
▼
┌───────────────────┐
│  SMOTE Resampling │  ← Handles extreme class imbalance
│                   │    (fraud << legitimate transactions)
└────────┬──────────┘
│
▼
┌───────────────────┐
│   Model Training  │  ← Logistic Regression (default)
│                   │    Supports: RandomForest, XGBoost
└────────┬──────────┘
│
▼
┌───────────────────┐
│ Threshold Tuning  │  ← PR curve analysis, optimal @ 0.8370
│                   │    Precision=0.955, Recall=0.991
└────────┬──────────┘
│
▼
┌───────────────────┐
│  MLflow Logging   │  ← Parameters, metrics, confusion matrix,
│                   │    PR curve, serialized model artifact
└───────────────────┘

### Feature Engineering Details

| Feature | Type | Description |
|---------|------|-------------|
| `category_x_payment` | Interaction | Category × PaymentMethod cross feature |
| `age_ratio` | Ratio | `paymentMethodAgeDays / accountAgeDays` |
| `account_age_bin` | Binning | new / medium / old buckets |
| `time_of_day` | Temporal | localTime → morning/afternoon/evening/night |

---

## 4. Tech Stack

### Core ML & Data
| Library | Version | Purpose |
|---------|---------|---------|
| scikit-learn | latest | Model building, preprocessing, metrics |
| imbalanced-learn | latest | SMOTE for class imbalance |
| pandas | latest | Data manipulation |
| numpy | latest | Numerical operations |
| xgboost | latest | Gradient boosted trees (optional model) |

### MLOps & Deployment
| Tool | Purpose |
|------|---------|
| MLflow | Experiment tracking, model registry, artifact storage |
| FastAPI | High-performance REST API serving |
| Streamlit | Interactive prediction UI |
| Docker | Microservice containerization |
| Kubernetes (Minikube) | Local container orchestration & scaling |
| Prometheus | Time-series metrics collection |
| Grafana | Metrics visualization & alerting dashboards |

---

## 5. Directory Structure
fraud-detection-mlops/
│
├── API/                           # FastAPI microservice
│   ├── main.py                    # API entry point, routes
│   ├── schemas.py                 # Pydantic request/response models
│   ├── services.py                # Prediction service logic
│   └── mlruns/                    # MLflow experiment logs
│
├── Data/                          # Transaction datasets
│   ├── payment_fraud.csv          # Training data
│   └── combined_holdout.csv       # Evaluation datasets
│
├── Images/                        # Architecture diagrams & screenshots
│   ├── MLOps_Architecture/
│   ├── Model_Architecture/
│   ├── Docker/
│   ├── FastAPI/
│   ├── Grafana/
│   └── Prometheus/
│
├── K8s/                           # Kubernetes manifests
│   ├── fraud-api-deployment.yaml  # FastAPI deployment spec
│   ├── fraud-api-service.yaml     # FastAPI service spec
│   ├── grafana-deployment.yaml    # Grafana deployment
│   └── prometheus-deployment.yaml # Prometheus deployment
│
├── Notebooks/                     # Jupyter Notebooks
│   ├── EDA.ipynb                  # Exploratory data analysis
│   ├── training_model.ipynb       # Model training walkthrough
│   ├── test_files.ipynb           # Testing and validation
│   └── artifacts/                 # Saved model artifacts
│       ├── confusion_matrix.png
│       ├── pr_curve.png
│       └── fraud_pipeline_deployed.pkl
│
├── Pages/                         # Streamlit multi-page app
│   ├── home.py                    # Prediction page
│   ├── about_model.py             # Model explanation page
│   ├── metrics_page.py            # Performance metrics page
│   └── about_me.py                # Developer info page
│
├── Src/                           # Core ML source code
│   ├── model.py                   # FraudPipeline class definition
│   ├── utils.py                   # Helper functions
│   ├── config.py                  # Global configuration
│   └── artifacts/                 # MLflow model output
│
├── app.py                         # Streamlit entry point
├── Dockerfile                     # Docker image definition
├── requirements.txt               # Python dependencies
├── .dockerignore                  # Docker build exclusions
└── .gitignore                     # Git exclusions

---

## 6. Setup & Installation

### Prerequisites
- Python 3.10+
- Docker Desktop
- Minikube + kubectl
- Git

### Local Development
```bash
# Clone the repository
git clone https://github.com/Shubhamharanale7/fraud-detection-mlops.git
cd fraud-detection-mlops

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## 7. Running Services

### Streamlit UI
```bash
streamlit run app.py
# Access: http://localhost:8501
```

### FastAPI Server
```bash
cd API
uvicorn main:app --reload --host 0.0.0.0 --port 8000
# Swagger docs: http://localhost:8000/docs
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Submit transaction, get fraud prediction |
| GET | `/health` | Service health check |
| GET | `/metrics` | Prometheus metrics scrape endpoint |

### Sample API Request
```json
POST /predict
{
  "category": "electronics",
  "payment_method": "credit_card",
  "account_age_days": 120,
  "payment_method_age_days": 45,
  "transaction_amount": 2500.00,
  "local_time": 14
}
```

### Sample API Response
```json
{
  "prediction": "FRAUD",
  "confidence": 0.94,
  "threshold_used": 0.837,
  "risk_score": 0.96
}
```

---

## 8. MLflow Experiment Tracking
```bash
mlflow ui
# Access: http://127.0.0.1:5000
```

### What Gets Tracked

| Category | Details |
|----------|---------|
| Parameters | Pipeline steps, resampling method, model type, threshold |
| Metrics | Accuracy, Precision, Recall, F1-score, PR-AUC, ROC-AUC |
| Artifacts | PR curve plot, confusion matrix, serialized model (.pkl) |
| Tags | Dataset version, experiment name, run description |

---

## 9. Docker & Kubernetes Deployment

### Docker
```bash
# Build images
docker build -t fraud-streamlit -f Dockerfile .
docker build -t fraud-fastapi -f Dockerfile ./API

# Run containers
docker run -p 8501:8501 fraud-streamlit
docker run -p 8000:8000 fraud-fastapi
```

### Kubernetes (Minikube)
```bash
# Start cluster
minikube start --driver=docker

# Deploy all services
kubectl apply -f K8s/fraud-api-deployment.yaml
kubectl apply -f K8s/fraud-api-service.yaml
kubectl apply -f K8s/prometheus-deployment.yaml
kubectl apply -f K8s/grafana-deployment.yaml

# Verify pods
kubectl get pods --all-namespaces

# Access services
minikube service fraud-api-service
minikube service prometheus -n monitoring
minikube service grafana -n monitoring
```

---

## 10. Monitoring Stack

### Prometheus
- Scrapes FastAPI `/metrics` endpoint every 15 seconds
- Tracks: request count, response latency, error rate, prediction distribution
- Access: `http://localhost:9090`

### Grafana
- Connects to Prometheus as data source
- Pre-built dashboards for:
  - API request throughput
  - Fraud vs legitimate prediction ratio
  - Model response time (P50, P95, P99)
  - System resource usage (CPU, memory)
- Access: `http://localhost:3000`

### Key Metrics Monitored

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `fraud_predictions_total` | Total fraud predictions | — |
| `api_request_latency_seconds` | Response time histogram | P95 > 500ms |
| `model_error_rate` | Failed predictions / total | > 1% |
| `fraud_rate_ratio` | Fraud / total predictions | Sudden spike |

---

## 11. Model Details & Results

### Model Configuration
| Parameter | Value |
|-----------|-------|
| Algorithm | Logistic Regression (default) |
| Resampling | SMOTE |
| Optimal Threshold | 0.8370 |
| Precision @ threshold | 95.5% |
| Recall @ threshold | 99.1% |

### Hold-out Evaluation Results

| Dataset | Accuracy | Recall | Precision | Notes |
|---------|----------|--------|-----------|-------|
| Hold-out A | 97% | 100% | 25% | Highly imbalanced |
| Hold-out B | 99% | 100% | 50% | Moderately imbalanced |
| Hold-out C | 98% | 98% | 98% | Balanced distribution |

### Why High Recall Matters
In fraud detection, **missing a fraud (false negative) is far more costly**
than a false alarm (false positive). The threshold is tuned to maximize
recall while keeping precision acceptable for business operations.

---

## 12. Future Roadmap

- [ ] CI/CD pipeline with GitHub Actions (auto-train on new data)
- [ ] Cloud deployment on AWS EKS / GCP GKE
- [ ] Real-time streaming predictions with Apache Kafka
- [ ] Model explainability with SHAP / LIME
- [ ] A/B testing framework for model versions
- [ ] Automated model retraining on data drift detection
- [ ] MLflow model registry with staging/production promotion

---

## 13. Contact

**Shubham Haranale**

📩 [LinkedIn](https://www.linkedin.com/in/shubhamharanale7)  
📧 shubhaminfosoft7@gmail.com  
🐙 [GitHub](https://github.com/Shubhamharanale7)

Your feedback helps me grow. Let's connect and build something great!
