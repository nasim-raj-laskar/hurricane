from airflow import DAG                                                        # type: ignore
from airflow.operators.python import PythonOperator                            # type: ignore
from airflow.hooks.base import BaseHook                                        # type: ignore
from datetime import datetime, timedelta                            
import os, zipfile, boto3                                                            

# CONFIG
S3_BUCKET = 'hurricane-damage-data'
DATA_DIR = '/tmp/data'
LOCAL_ZIP = os.path.join(DATA_DIR, 'dataset.zip')
IMG_SIZE = (128, 128)
BATCH = 16

default_args = {
    'owner': 'hurricane-team',
    'start_date': datetime(2025, 10, 26),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

#task-1
def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)

    conn = BaseHook.get_connection('aws_default')

    s3 = boto3.client(
        's3',
        aws_access_key_id=conn.login,
        aws_secret_access_key=conn.password,
        region_name=conn.extra_dejson.get('region_name', 'eu-north-1')
    )

    s3.download_file(S3_BUCKET, 'dataset.zip', LOCAL_ZIP)
    with zipfile.ZipFile(LOCAL_ZIP, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)

    return os.path.join(DATA_DIR, 'dataset')


#task-2
def load_datasets(**context):
    root = context["task_instance"].xcom_pull(task_ids='download_data')

    dirs = {
        "train": os.path.join(root, "train_another"),
        "val": os.path.join(root, "validation_another"),
        "test": os.path.join(root, "test")
    }

    return dirs


#task-3
def build_and_train(**context):
    import mlflow
    import tensorflow as tf
    from tensorflow.keras import Sequential, Input                                                                                         #type:ignore
    from tensorflow.keras.layers import (Dense, Dropout, Conv2D, MaxPooling2D,SeparableConv2D, GlobalAveragePooling2D, Rescaling)          #type:ignore
    import sys
    import warnings
    warnings.filterwarnings("ignore")

    conn = BaseHook.get_connection("mlflow_dagshub")

    os.environ["MLFLOW_TRACKING_USERNAME"] = conn.login
    os.environ["MLFLOW_TRACKING_PASSWORD"] = conn.password
    os.environ["MLFLOW_TRACKING_URI"] = conn.extra_dejson.get(
        "tracking_uri", "https://dagshub.com/nasim-raj-laskar/hurricane.mlflow"
    )

    print("[INFO] Using Dagshub MLflow Auth:")
    print(" - User:", os.getenv("MLFLOW_TRACKING_USERNAME"))
    print(" - URI:", os.getenv("MLFLOW_TRACKING_URI"))

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("hurricane_damage_training_v5")

    #sanity check
    print("MLFLOW_TRACKING_USERNAME:", os.getenv("MLFLOW_TRACKING_USERNAME"))
    print("MLFLOW_TRACKING_PASSWORD:", os.getenv("MLFLOW_TRACKING_PASSWORD")[:4], "***")
    print("MLFLOW_TRACKING_URI:", os.getenv("MLFLOW_TRACKING_URI"))

    dirs = context['task_instance'].xcom_pull(task_ids='load_datasets')

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dirs["train"], image_size=IMG_SIZE, batch_size=BATCH
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dirs["val"], image_size=IMG_SIZE, batch_size=BATCH
    )

    #Model
    model = Sequential([
        Input(shape=(128, 128, 3)),
        Rescaling(1./255),
        Conv2D(8, 3, activation='relu', padding='same'),
        MaxPooling2D(),
        Conv2D(16, 3, activation='relu', padding='same'),
        MaxPooling2D(),
        SeparableConv2D(32, 3, activation='relu', padding='same'),
        GlobalAveragePooling2D(),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    #MLflow 
    with mlflow.start_run(run_name="hurricane_training") as run:
        history = model.fit(train_ds, validation_data=val_ds, epochs=2)

        # Log to MLflow
        mlflow.log_params({
            "epochs": 2,
            "batch_size": BATCH,
            "img_size": IMG_SIZE,
            "optimizer": "Adam",
            "learning_rate": 0.001
        })
        mlflow.log_metric("train_accuracy", history.history['accuracy'][-1] * 100)
        mlflow.log_metric("val_accuracy", history.history['val_accuracy'][-1] * 100)
        mlflow.log_metric("train_loss", history.history['loss'][-1] * 100)
        mlflow.log_metric("val_loss", history.history['val_loss'][-1] * 100)

        
        # Local backup logs
        print("=== ML TRAINING METRICS ===")
        for epoch in range(len(history.history['accuracy'])):
            metrics = {
                'epoch': epoch + 1,
                'train_accuracy': round(history.history['accuracy'][epoch] * 100, 2),
                'val_accuracy': round(history.history['val_accuracy'][epoch] * 100, 2),
                'train_loss': round(history.history['loss'][epoch], 4),
                'val_loss': round(history.history['val_loss'][epoch], 4)
            }
            print(f"TRAINING_METRIC: {metrics}")
        print("=== END METRICS ===")

        model.save('/tmp/trained_model.h5')
        mlflow.log_artifact('/tmp/trained_model.h5', artifact_path="model")

        # Push run_id for next tasks
        context['task_instance'].xcom_push(key='run_id', value=run.info.run_id)

    return True


#task-4
def evaluate_model(**context):
    import tensorflow as tf
    dirs = context['task_instance'].xcom_pull(task_ids='load_datasets')
    model = tf.keras.models.load_model('/tmp/trained_model.h5')

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dirs["test"], image_size=IMG_SIZE, batch_size=BATCH
    ).cache().prefetch(tf.data.AUTOTUNE)

    loss, acc = model.evaluate(test_ds)
    return {'accuracy': acc, 'loss': loss}


#task-5
def save_model_s3(**context):
    import boto3, mlflow, os
    from datetime import datetime
    from airflow.hooks.base import BaseHook #type:ignore

    run_id = context['task_instance'].xcom_pull(task_ids='build_and_train', key='run_id')

    mlflow_conn = BaseHook.get_connection('mlflow_dagshub')
    os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_conn.login
    os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_conn.password
    mlflow.set_tracking_uri(mlflow_conn.extra_dejson['tracking_uri'])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_model = '/tmp/trained_model.h5'
    renamed_path = f"/tmp/hurricane_{timestamp}.h5"

    if os.path.exists(tmp_model):
        os.rename(tmp_model, renamed_path)
    else:
        raise FileNotFoundError(f"{tmp_model} not found before upload!")

    s3 = boto3.client('s3')
    s3.upload_file(renamed_path, S3_BUCKET, f"models/hurricane_{timestamp}.h5")

    s3_uri = f"s3://{S3_BUCKET}/models/hurricane_{timestamp}.h5"

    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("s3_model_path", s3_uri)
        mlflow.log_artifact(renamed_path)

    # Cleanup
    os.remove(renamed_path)
    return s3_uri



# DAG DEFINITION
dag = DAG(
    'hurricane_damage_training_v5',
    default_args=default_args,
    schedule=None,
    catchup=False,
    tags=['ml', 'tensorflow', 'dagshub', 'mlflow']
)

t1 = PythonOperator(task_id='download_data', python_callable=download_data, dag=dag)
t2 = PythonOperator(task_id='load_datasets', python_callable=load_datasets, dag=dag)
t3 = PythonOperator(task_id='build_and_train', python_callable=build_and_train, dag=dag)
t4 = PythonOperator(task_id='evaluate_model', python_callable=evaluate_model, dag=dag)
t5 = PythonOperator(task_id='save_model_s3', python_callable=save_model_s3, dag=dag)

t1 >> t2 >> t3 >> t4 >> t5
