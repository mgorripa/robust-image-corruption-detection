# Robust Image Corruption Detection

PyTorch-based image corruption detection model (clean vs corrupted) trained on CIFAR-10 and served via a FastAPI microservice with Docker and Kubernetes deployment.
---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Training (Step-by-Step)](#training-step-by-step)
- [Running the API (FastAPI)](#running-the-api-fastapi)
- [Docker (Build and Run)](#docker-build-and-run)
- [Kubernetes (Deploy and Autoscale)](#kubernetes-deploy-and-autoscale)
- [Future Extensions](#future-extensions)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Project Overview

This project detects whether an input image is **clean** or **corrupted** (noise, blur, compression, occlusion, etc.). It has three layers:

1. **Model training (PyTorch + CIFAR-10)**  
   - Wraps CIFAR-10 into a binary classification task: clean vs corrupted.
   - Applies random corruptions (Gaussian noise, blur, JPEG-style compression, brightness changes, occlusion).
   - Trains a ResNet18-based classifier and evaluates validation accuracy.

2. **Inference & API (FastAPI)**  
   - Loads the trained model and exposes a `/predict` endpoint.
   - Accepts an image upload and returns:
     ```json
     {
       "label": "clean" or "corrupted",
       "confidence": 0.95,
       "probs": {
         "clean": 0.05,
         "corrupted": 0.95
       }
     }
     ```

3. **Deployment (Docker + Kubernetes)**  
   - Dockerfile to build a container image for the service.
   - Kubernetes manifests (`Deployment`, `Service`, `HPA`) to run the service in a cluster with basic autoscaling.

This project showcases end-to-end ML system skills: **model training, robustness analysis, API serving, containerization, and cluster deployment.**

---

## Repository Structure

```text
robust-image-corruption-detection/
├── README.md
├── requirements.txt
├── src
│   ├── dataset.py       # CIFAR-10 + corruption dataset wrapper
│   ├── model.py         # ResNet18 model definition
│   ├── train.py         # Training script
│   └── infer.py         # Inference helper class
├── service
│   ├── main.py          # FastAPI app
│   └── Dockerfile       # Container for the API
├── artifacts
│   └── .gitkeep         # Trained model saved here as model.pt
└── k8s
    ├── deployment.yaml  # Kubernetes Deployment
    ├── service.yaml     # Kubernetes Service
    └── hpa.yaml         # Horizontal Pod Autoscaler
```
---

## Prerequisites

- Python 3.10+ recommended
- pip and virtualenv (or `python -m venv`)
- Docker (for containerization)
- Kubernetes cluster + `kubectl` (kind/minikube or cloud cluster) for the K8s section

---

## Setup

```bash
git clone https://github.com/mgorripa/robust-image-corruption-detection.git
cd robust-image-corruption-detection

python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

---

## Training (Step-by-Step)

These steps train the corruption-detection model and save weights to `artifacts/model.pt`.

1) **Activate environment and run training:**
```bash
cd src
python train.py
```

2) **What happens:**
- CIFAR-10 downloads to `./data` (if not present).
- `CIFAR10CorruptionDataset` yields either clean or randomly corrupted samples.
- ResNet18 is trained for `num_epochs` (default: 10).
- Validation metrics are printed each epoch.
- The best model (by validation accuracy) is saved to `../artifacts/model.pt`.

3) **Verify artifact:**
```bash
ls ../artifacts
# should contain: model.pt
```

> Tip: If you have a GPU, `torch.cuda.is_available()` will be used automatically for training/inference.

---

## Running the API (FastAPI)

1) **From repo root (ensure venv active):**
```bash
uvicorn service.main:app --reload --port 8000
```

2) **Health check:**
```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

3) **Predict on an image:**
```bash
curl -X POST "http://localhost:8000/predict"   -H "accept: application/json"   -H "Content-Type: multipart/form-data"   -F "file=@path/to/your/image.jpg"
```

**Example response:**
```json
{
  "label": "corrupted",
  "confidence": 0.94,
  "probs": {
    "clean": 0.06,
    "corrupted": 0.94
  }
}
```

---

## Docker (Build and Run)

> Ensure `artifacts/model.pt` exists (from training).

### Build
```bash
cd service
docker build -t mgorripa/image-corruption-service:latest .
```

### Run
```bash
docker run -p 8000:8000 mgorripa/image-corruption-service:latest
```

### Test inside container
```bash
curl http://localhost:8000/health
curl -X POST "http://localhost:8000/predict"   -H "accept: application/json"   -H "Content-Type: multipart/form-data"   -F "file=@path/to/your/image.jpg"
```

> If you prefer a different image name, change the tag accordingly.

---

## Kubernetes (Deploy and Autoscale)

> You need a running cluster and `kubectl` configured. If using minikube:
> - `minikube start`
> - Push the image to a registry accessible by the cluster (e.g., Docker Hub).

### 1) Push image to a registry

```bash
# Example using Docker Hub
docker tag mgorripa/image-corruption-service:latest <your-dockerhub-username>/image-corruption-service:latest
docker push <your-dockerhub-username>/image-corruption-service:latest
```

### 2) Update image in k8s manifest

In `k8s/deployment.yaml`, set:
```yaml
containers:
  - name: image-corruption-service
    image: <your-dockerhub-username>/image-corruption-service:latest
```

### 3) Apply manifests
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

### 4) Verify
```bash
kubectl get pods
kubectl get svc image-corruption-service
kubectl get hpa
```

If using minikube:
```bash
minikube service image-corruption-service --url
```
Use the printed URL to hit `/health` and `/predict`.

---

## Future Extensions

1) **Multi-class corruption detection**  
   - Predict specific corruption types (`clean`, `noise`, `blur`, `compression`, `brightness`, `occlusion`).
   - Change the dataset to return corruption class labels and set `num_classes > 2`.
   - Update final FC layer in `model.py`, retrain, and update API label mapping.

2) **Robustness analysis**  
   - Add evaluation scripts to vary corruption severity (e.g., 1–3).  
   - Produce robustness curves (accuracy vs severity) and confusion matrices per corruption type.  
   - Save plots to a `reports/` directory and link in README.

3) **Explainability**  
   - Integrate Grad-CAM or similar to visualize regions influencing predictions.  
   - Provide example heatmaps for selected clean vs corrupted images.

4) **Observability**  
   - Add Prometheus metrics to FastAPI (request counts, latency distributions, error rates, counts per predicted label).  
   - Provide a Grafana dashboard JSON in `observability/` with basic panels.

5) **GPU & larger models**  
   - Build a CUDA-enabled Docker image (nvidia/cuda base) for GPU inference.  
   - Experiment with ResNet34/50 or ViT; compare accuracy vs latency/size.  
   - Document trade-offs (accuracy, throughput, memory).

6) **Front-end**  
   - Simple web UI to drag-and-drop an image and see predictions and optional Grad-CAM overlays.

---

## Troubleshooting

- **`model.pt` not found**  
  - Ensure training completed and `artifacts/model.pt` exists before starting the API or building Docker.

- **CUDA not used**  
  - If no GPU is available, the code runs on CPU. Confirm with `print(torch.cuda.is_available())`.

- **Image upload errors**  
  - Ensure you send a valid image (content-type `image/*`). The API converts to RGB internally.

- **Kubernetes image pull errors**  
  - Verify the image name/tag in `deployment.yaml` and that it’s pushed to a registry accessible by your cluster.

- **Port conflicts**  
  - If `8000` is taken locally or in Docker, change the exposed and mapped port accordingly.

---

## License

MIT — see `LICENSE`.
