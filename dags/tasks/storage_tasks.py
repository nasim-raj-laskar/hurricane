"""Model storage tasks"""

import os
from datetime import datetime
from airflow.hooks.base import BaseHook  # type: ignore

from dags.config import S3_BUCKET


def save_model_s3(**context):
    """Save trained model to S3"""

    import boto3
    import mlflow

    ti = context["task_instance"]
    run_id = ti.xcom_pull(task_ids="build_and_train", key="run_id")

    # ---------- MLflow setup ----------
    mlflow_conn = BaseHook.get_connection("mlflow_dagshub")

    os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_conn.login
    os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_conn.password
    mlflow.set_tracking_uri(mlflow_conn.extra_dejson["tracking_uri"])

    # ---------- AWS S3 client ----------
    aws_conn = BaseHook.get_connection("aws_default")

    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_conn.login,
        aws_secret_access_key=aws_conn.password,
        region_name=aws_conn.extra_dejson.get("region_name", "eu-north-1"),
    )

    # ---------- Prepare model ----------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_model = "/tmp/trained_model.h5"
    renamed_path = f"/tmp/hurricane_{timestamp}.h5"

    if not os.path.exists(tmp_model):
        raise FileNotFoundError(f"{tmp_model} not found before upload!")

    os.rename(tmp_model, renamed_path)

    # ---------- Upload to S3 ----------
    s3_key = f"models/hurricane_{timestamp}.h5"
    s3.upload_file(renamed_path, S3_BUCKET, s3_key)

    s3_uri = f"s3://{S3_BUCKET}/{s3_key}"

    # ---------- Log to MLflow ----------
    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("s3_model_path", s3_uri)
        mlflow.log_artifact(renamed_path)

    os.remove(renamed_path)

    return s3_uri
