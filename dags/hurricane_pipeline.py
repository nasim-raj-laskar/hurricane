"""Hurricane Damage Training Pipeline DAG"""
from airflow import DAG  # type: ignore
from airflow.operators.python import PythonOperator  # type: ignore
from dags.config import DEFAULT_ARGS
from dags.tasks.data_tasks import download_data, load_datasets
from dags.tasks.training_tasks import build_and_train
from dags.tasks.evaluation_tasks import evaluate_model
from dags.tasks.storage_tasks import save_model_s3



# DAG Definition
dag = DAG(
    'hurricane_damage_training_mod',
    default_args=DEFAULT_ARGS,
    schedule=None,
    catchup=False,
    tags=['ml', 'tensorflow', 'dagshub', 'mlflow']
)

# Task Definitions
t1 = PythonOperator(task_id='download_data', python_callable=download_data, dag=dag)
t2 = PythonOperator(task_id='load_datasets', python_callable=load_datasets, dag=dag)
t3 = PythonOperator(task_id='build_and_train', python_callable=build_and_train, dag=dag)
t4 = PythonOperator(task_id='evaluate_model', python_callable=evaluate_model, dag=dag)
t5 = PythonOperator(task_id='save_model_s3', python_callable=save_model_s3, dag=dag)

# Task Dependencies
t1 >> t2 >> t3 >> t4 >> t5
