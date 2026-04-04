import mlflow.pyfunc
import pandas as pd
import numpy as np
import mlflow
from mlflow import MlflowClient

# mlflow.set_tracking_uri('file:./Notebooks/mlruns')
mlflow.set_tracking_uri('file:./API/mlruns')

client = MlflowClient()

# Set model version alias
model_name = "FraudDetectionPipeline"
model_version_alias = "champion"

# Get information about the model
model_info = client.get_model_version_by_alias(model_name, model_version_alias)
model_tags = model_info.tags
print(model_tags)

# Get the model version using a model URI
MLFLOW_MODEL_URI = f"models:/{model_name}@{model_version_alias}"
model = mlflow.pyfunc.load_model(MLFLOW_MODEL_URI)

def predict_fraud(data: dict):
    df = pd.DataFrame([data])
    df = df.where(pd.notnull(df), np.nan)  # converts None to NaN
    preds = model.predict(df)
    return int(preds[0])
