"""Model storage tasks"""
import os
import boto3
import mlflow
from datetime import datetime
from airflow.hooks.base import BaseHook  # type: ignore
from config import S3_BUCKET


def save_model_s3(**context):
    """Save trained model to S3"""
    run_id = context['task_instance'].xcom_pull(task_ids='build_and_train', key='run_id')

    # Setup MLflow
    mlflow_conn = BaseHook.get_connection('mlflow_dagshub')
    os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_conn.login
    os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_conn.password
    mlflow.set_tracking_uri(mlflow_conn.extra_dejson['tracking_uri'])

    # Get AWS credentials from Airflow Connection
    aws_conn = BaseHook.get_connection('aws_default')
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_conn.login,
        aws_secret_access_key=aws_conn.password,
        region_name=aws_conn.extra_dejson.get('region_name', 'eu-north-1')
    )

    # Prepare model file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_model = '/tmp/trained_model.h5'
    renamed_path = f"/tmp/hurricane_{timestamp}.h5"

    if os.path.exists(tmp_model):
        os.rename(tmp_model, renamed_path)
    else:
        raise FileNotFoundError(f"{tmp_model} not found before upload!")

    # Upload to S3
    s3.upload_file(renamed_path, S3_BUCKET, f"models/hurricane_{timestamp}.h5")

    s3_uri = f"s3://{S3_BUCKET}/models/hurricane_{timestamp}.h5"

    # Log to MLflow
    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("s3_model_path", s3_uri)
        mlflow.log_artifact(renamed_path)

    # Cleanup
    os.remove(renamed_path)
    return s3_uri
