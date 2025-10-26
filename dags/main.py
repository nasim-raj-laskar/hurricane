from airflow import DAG #type:ignore
from airflow.operators.python import PythonOperator #type:ignore
from airflow.hooks.base import BaseHook #type:ignore
from datetime import datetime, timedelta
import os
import shutil
import boto3
import zipfile
import tensorflow as tf
from tensorflow.keras import Sequential, Input #type:ignore
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, SeparableConv2D, GlobalAveragePooling2D, Rescaling #type:ignore
from tensorflow.keras.callbacks import EarlyStopping #type:ignore
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.run_functions_eagerly(True)

# Configuration
S3_BUCKET = 'hurricane-damage-data'
S3_KEY = 'dataset.zip'
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

def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)

    conn = BaseHook.get_connection('aws_default')

    s3 = boto3.client(
        's3',
        aws_access_key_id=conn.login,
        aws_secret_access_key=conn.password,
        region_name=conn.extra_dejson.get('region_name', 'eu-north-1')
    )

    s3.download_file(S3_BUCKET, S3_KEY, LOCAL_ZIP)

    with zipfile.ZipFile(LOCAL_ZIP, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)

    extracted_root = os.path.join(DATA_DIR, 'dataset')
    return extracted_root


def load_datasets(**context):
    root = context["task_instance"].xcom_pull(task_ids='download_data')

    dirs = {
        "train": os.path.join(root, "train_another"),
        "val": os.path.join(root, "validation_another"),
        "test": os.path.join(root, "test")
    }

    # Load datasets with caching & prefetching for performance
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dirs["train"], image_size=IMG_SIZE, batch_size=BATCH
    ).cache().prefetch(tf.data.AUTOTUNE)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dirs["val"], image_size=IMG_SIZE, batch_size=BATCH
    ).cache().prefetch(tf.data.AUTOTUNE)

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dirs["test"], image_size=IMG_SIZE, batch_size=BATCH
    ).cache().prefetch(tf.data.AUTOTUNE)

    # Save preloaded dataset paths in XCom
    return dirs


def build_and_train(**context):
    dirs = context['task_instance'].xcom_pull(task_ids='load_datasets')

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
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy'],
        run_eagerly=True
    )

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dirs["train"], image_size=IMG_SIZE, batch_size=BATCH
    ).cache().prefetch(tf.data.AUTOTUNE)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dirs["val"], image_size=IMG_SIZE, batch_size=BATCH
    ).cache().prefetch(tf.data.AUTOTUNE)

    es = EarlyStopping(patience=3, restore_best_weights=True)

    model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=[es],verbose=2)
    model.save('/tmp/trained_model.h5')

    return '/tmp/trained_model.h5'


def evaluate_model(**context):
    dirs = context['task_instance'].xcom_pull(task_ids='load_datasets')

    model = tf.keras.models.load_model('/tmp/trained_model.h5')
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dirs["test"], image_size=IMG_SIZE, batch_size=BATCH
    ).cache().prefetch(tf.data.AUTOTUNE)

    loss, acc = model.evaluate(test_ds)
    
    # Push metrics to XCom for UI visibility
    return {'accuracy': acc, 'loss': loss}


def save_model_s3():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"hurricane_{timestamp}.h5"

    os.rename('/tmp/trained_model.h5', model_path)

    s3 = boto3.client('s3')
    s3.upload_file(model_path, S3_BUCKET, f"models/{model_path}")
    os.remove(model_path)

    # Auto-remove /tmp/data to free space
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)

    return f"s3://{S3_BUCKET}/models/{model_path}"


dag = DAG(
    'hurricane_damage_training_v3',
    default_args=default_args,
    description='Train hurricane damage detection model with caching, cleanup, and metrics',
    schedule=None,
    catchup=False
)

download_task = PythonOperator(task_id='download_data', python_callable=download_data, dag=dag)
load_task = PythonOperator(task_id='load_datasets', python_callable=load_datasets, dag=dag)
train_task = PythonOperator(task_id='build_and_train', python_callable=build_and_train, dag=dag)
evaluate_task = PythonOperator(task_id='evaluate_model', python_callable=evaluate_model, dag=dag)
save_task = PythonOperator(task_id='save_model_s3', python_callable=save_model_s3, dag=dag)

download_task >> load_task >> train_task >> evaluate_task >> save_task
