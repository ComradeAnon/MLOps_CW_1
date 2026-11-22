from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

project = Path(__file__).resolve().parent
model = joblib.load(project / "model.joblib")

mlflow.set_tracking_uri(f"sqlite:///{project / 'mlflow.db'}")
mlflow.set_experiment("telco_churn")

with mlflow.start_run(run_name="random_forest") as run:
    mlflow.log_params(model.named_steps["randomforestclassifier"].get_params())
    mlflow.log_metrics(model.validation_metrics_)
    mlflow.sklearn.log_model(model, name="churn_model", serialization_format="cloudpickle")

print(f"MLflow run: {run.info.run_id}")
