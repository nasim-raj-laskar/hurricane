from airflow import DAG                                                                                                         #type:ignore
from airflow.operators.python import PythonOperator                                                                             #type:ignore
from airflow.hooks.base import BaseHook                                                                                         #type:ignore
from datetime import datetime, timedelta                                                                                        #type:ignore
import os
import boto3
import zipfile
import tensorflow as tf
from tensorflow.keras import Sequential, Input                                                                                  #type:ignore
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, SeparableConv2D, GlobalAveragePooling2D, Rescaling    #type:ignore
from tensorflow.keras.callbacks import EarlyStopping                                                                            #type:ignore
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configuration
S3_BUCKET = 'hurricane-damage-data'
S3_KEY = 'dataset.zip'
DATA_DIR = '/tmp/data'
LOCAL_ZIP = os.path.join(DATA_DIR, 'dataset.zip')
IMG_SIZE = (128, 128)
BATCH = 32

default_args = {
    'owner': 'hurricane-team',
    'start_date': datetime(2025, 10, 26),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)

    conn = BaseHook.get_connection('aws_default')

    s3 = boto3.client('s3',
                      aws_access_key_id=conn.login,
                      aws_secret_access_key=conn.password,
                      region_name=conn.extra_dejson.get('region_name', 'eu-north-1'))
    
    s3.download_file(S3_BUCKET, S3_KEY, LOCAL_ZIP)
    
    with zipfile.ZipFile(LOCAL_ZIP, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    
    extracted_root = os.path.join(DATA_DIR, 'dataset')
    return {
        "train": os.path.join(extracted_root, "train_another"),
        "val": os.path.join(extracted_root, "validation_another"),
        "test": os.path.join(extracted_root, "test")
    }

def load_datasets(**context):
    dirs = context['task_instance'].xcom_pull(task_ids='download_data')
    
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dirs["train"], image_size=IMG_SIZE, batch_size=BATCH)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dirs["val"], image_size=IMG_SIZE, batch_size=BATCH)
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dirs["test"], image_size=IMG_SIZE, batch_size=BATCH)
    
    return dirs

def build_model():
    model = Sequential([
        Input(shape=(128, 128, 3)),
        Rescaling(1./255),
        Conv2D(8, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(),
        Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(),
        SeparableConv2D(32, 3, activation='relu', padding='same'),
        GlobalAveragePooling2D(),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    model.save('/tmp/untrained_model.h5')
    return '/tmp/untrained_model.h5'

def train_model(**context):
    dirs = context['task_instance'].xcom_pull(task_ids='load_datasets')
    
    model = tf.keras.models.load_model('/tmp/untrained_model.h5')
    
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dirs["train"], image_size=IMG_SIZE, batch_size=BATCH)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dirs["val"], image_size=IMG_SIZE, batch_size=BATCH)
    
    es = EarlyStopping(patience=3, restore_best_weights=True)
    model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=[es])
    
    model.save('/tmp/trained_model.h5')
    return '/tmp/trained_model.h5'

def evaluate_model(**context):
    dirs = context['task_instance'].xcom_pull(task_ids='load_datasets')
    
    model = tf.keras.models.load_model('/tmp/trained_model.h5')
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dirs["test"], image_size=IMG_SIZE, batch_size=BATCH)
    
    loss, acc = model.evaluate(test_ds)
    return {'accuracy': acc, 'loss': loss}

def save_model_s3(**context):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"hurricane_{timestamp}.h5"
    
    os.rename('/tmp/trained_model.h5', model_path)
    
    s3 = boto3.client('s3')
    s3.upload_file(model_path, S3_BUCKET, f"models/{model_path}")
    os.remove(model_path)
    
    return f"s3://{S3_BUCKET}/models/{model_path}"

dag = DAG(
    'hurricane_damage_training',
    default_args=default_args,
    description='Train hurricane damage detection model',
    schedule_interval='@weekly',
    catchup=False
)

download_task = PythonOperator(
    task_id='download_data',
    python_callable=download_data,
    dag=dag
)

load_task = PythonOperator(
    task_id='load_datasets',
    python_callable=load_datasets,
    dag=dag
)

build_task = PythonOperator(
    task_id='build_model',
    python_callable=build_model,
    dag=dag
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

evaluate_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag
)

save_task = PythonOperator(
    task_id='save_model_s3',
    python_callable=save_model_s3,
    dag=dag
)

download_task >> load_task >> build_task >> train_task >> evaluate_task >> save_task