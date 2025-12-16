"""Data download and loading tasks"""

import os
import zipfile
from airflow.hooks.base import BaseHook  # type: ignore
from dags.config import S3_BUCKET, DATA_DIR, LOCAL_ZIP


def download_data():
    """Download dataset from S3 and extract"""
    import boto3

    os.makedirs(DATA_DIR, exist_ok=True)

    conn = BaseHook.get_connection("aws_default")

    s3 = boto3.client(
        "s3",
        aws_access_key_id=conn.login,
        aws_secret_access_key=conn.password,
        region_name=conn.extra_dejson.get("region_name", "eu-north-1"),
    )

    s3.download_file(S3_BUCKET, "dataset.zip", LOCAL_ZIP)

    with zipfile.ZipFile(LOCAL_ZIP, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

    return os.path.join(DATA_DIR, "dataset")


def load_datasets(**context):
    """Load train, validation, and test dataset paths"""

    ti = context["task_instance"]
    root = ti.xcom_pull(task_ids="download_data")

    dirs = {
        "train": os.path.join(root, "train_another"),
        "val": os.path.join(root, "validation_another"),
        "test": os.path.join(root, "test"),
    }

    return dirs
