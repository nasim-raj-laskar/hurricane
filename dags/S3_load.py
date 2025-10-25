from airflow import DAG # type: ignore
from airflow.operators.python import PythonOperator # type: ignore
from airflow.models import Connection # type: ignore
from airflow.hooks.base import BaseHook # type: ignore
import boto3
from datetime import datetime
import zipfile, os

S3_BUCKET = 'hurricane-damage-data'
S3_KEY = 'dataset.zip'
DATA_DIR = '/tmp/data'
LOCAL_ZIP = os.path.join(DATA_DIR, 'dataset.zip')

def download_and_prepare_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Fetch AWS creds from Airflow connection
    conn = BaseHook.get_connection('aws_default')
    s3 = boto3.client(
        's3',
        aws_access_key_id=conn.login,
        aws_secret_access_key=conn.password,
        region_name=conn.extra_dejson.get('region_name', 'eu-north-1')
    )
    
    # Download zip
    s3.download_file(S3_BUCKET, S3_KEY, LOCAL_ZIP)
    
    # Unzip
    with zipfile.ZipFile(LOCAL_ZIP, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    
    print(f"Dataset ready at {DATA_DIR}")
    print(f"Contents of {DATA_DIR}:")

    def print_folder_structure(base_dir):
        for root, dirs, files in os.walk(base_dir):
            level = root.replace(base_dir, "").count(os.sep)
            indent = "  " * level
            print(f"{indent}{os.path.basename(root)}/")
            sub_indent = "  " * (level + 1)
            for f in files:
                print(f"{sub_indent}{f}")

    # At the end of your DAG function
    print(f"\nFull folder structure of {DATA_DIR}:")
    print_folder_structure(DATA_DIR)


with DAG(
    dag_id='dataset_load_dag',
    start_date=datetime(2025, 10, 25),
    schedule=None,
    catchup=False,
    tags=['s3', 'dataset', 'training']
) as dag:

    prepare_dataset_task = PythonOperator(
        task_id='download_and_unzip_dataset',
        python_callable=download_and_prepare_dataset
    )

    prepare_dataset_task
