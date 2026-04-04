# ---------- Base Image ----------
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# ---------- Install dependencies ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Copy application code ----------
COPY API ./API
# MLflow local registry (if using local file storage)
RUN mkdir -p "/c:/Users/Asus/Downloads/Fraud_MLOps_Project/Notebooks/mlruns"
COPY Notebooks/mlruns /c:/Users/Asus/Downloads/Fraud_MLOps_Project/Notebooks/mlruns
# Artifacts if model paths are here
COPY Src/artifacts ./Src/artifacts         

# Expose FastAPI port
EXPOSE 8000

# ---------- Run FastAPI ----------
CMD ["uvicorn", "API.main:app", "--host", "0.0.0.0", "--port", "8000"]
