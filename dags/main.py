from airflow import DAG # type: ignore
from airflow.operators.python import PythonOperator # type: ignore
from datetime import datetime, timedelta
import os
import zipfile
import boto3
import tensorflow as tf
from tensorflow.keras import Sequential, Input # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, SeparableConv2D, GlobalAveragePooling2D, Rescaling # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Container-friendly paths
DATA_DIR = "/opt/airflow/dataset"
os.makedirs(DATA_DIR, exist_ok=True)

# S3 details
S3_BUCKET = "hurricane-damage-data"
S3_KEY = "dataset.zip"

# Constants
IMG_SIZE = (128, 128)
BATCH = 32
AUTOTUNE = tf.data.AUTOTUNE

default_args = {
    'owner': 'nasim',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id="hurricane_damage_cnn_pipeline",
    start_date=datetime(2025, 10, 25),
    schedule="@weekly",
    catchup=False,
    default_args=default_args,
    description="DAG for hurricane damage CNN model ",
) as dag:

    # Task 1: Fetch & unzip dataset from S3
    def fetch_data_from_s3():
        local_zip_path = os.path.join(DATA_DIR, "dataset.zip")
        print(f"Downloading s3://{S3_BUCKET}/{S3_KEY} ...")
        s3 = boto3.client("s3")
        s3.download_file(S3_BUCKET, S3_KEY, local_zip_path)

        print("Extracting dataset.zip ...")
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print(f"Dataset extracted to {DATA_DIR}")

    fetch_s3_task = PythonOperator(
        task_id="fetch_data_from_s3",
        python_callable=fetch_data_from_s3
    )

    # Task 2: Prepare tf.data datasets
    def prepare_datasets():
        train_dir = os.path.join(DATA_DIR, "train_another")
        val_dir = os.path.join(DATA_DIR, "validation_another")
        test_dir = os.path.join(DATA_DIR, "test")

        def augment(image, label):
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.1)
            scale = tf.random.uniform([], 0.9, 1.1)
            new_size = tf.cast(tf.convert_to_tensor(IMG_SIZE, dtype=tf.float32) * scale, tf.int32)
            image = tf.image.resize(image, new_size)
            image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE[0], IMG_SIZE[1])
            return image, label

        train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_dir, image_size=IMG_SIZE, batch_size=BATCH)
        train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE).cache().shuffle(1000).prefetch(AUTOTUNE)

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(val_dir, image_size=IMG_SIZE, batch_size=BATCH)
        val_ds = val_ds.cache().prefetch(AUTOTUNE)

        test_ds = tf.keras.preprocessing.image_dataset_from_directory(test_dir, image_size=IMG_SIZE, batch_size=BATCH)
        test_ds = test_ds.cache().prefetch(AUTOTUNE)

        # Save paths in XCom for next tasks
        return {"train": train_dir, "val": val_dir, "test": test_dir}

    prepare_task = PythonOperator(
        task_id="prepare_datasets",
        python_callable=prepare_datasets
    )

    # Task 3: Train model directly using tf.data
    def train_model(ti):
        paths = ti.xcom_pull(task_ids="prepare_datasets")
        train_dir, val_dir = paths["train"], paths["val"]

        def augment(image, label):
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.1)
            scale = tf.random.uniform([], 0.9, 1.1)
            new_size = tf.cast(tf.convert_to_tensor(IMG_SIZE, dtype=tf.float32) * scale, tf.int32)
            image = tf.image.resize(image, new_size)
            image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE[0], IMG_SIZE[1])
            return image, label

        train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_dir, image_size=IMG_SIZE, batch_size=BATCH)
        train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE).cache().shuffle(1000).prefetch(AUTOTUNE)

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(val_dir, image_size=IMG_SIZE, batch_size=BATCH)
        val_ds = val_ds.cache().prefetch(AUTOTUNE)

        model = Sequential([
            Input(shape=(128,128,3)),
            Rescaling(1./255),
            Conv2D(8,3,activation='relu',padding='same'), MaxPooling2D(),
            Conv2D(16,3,activation='relu',padding='same'), MaxPooling2D(),
            SeparableConv2D(32,3,activation='relu',padding='same'),
            GlobalAveragePooling2D(),
            Dense(32,activation='relu'), Dropout(0.2),
            Dense(1,activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        es = EarlyStopping(patience=3, restore_best_weights=True)

        model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=[es], verbose=1)

        model_path = os.path.join(DATA_DIR, "hurricane_model.h5")
        model.save(model_path)
        return model_path

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model
    )

    # Task 4: Evaluate & save model
    def evaluate_and_save_model(ti):
        paths = ti.xcom_pull(task_ids="prepare_datasets")
        test_dir = paths["test"]

        test_ds = tf.keras.preprocessing.image_dataset_from_directory(test_dir, image_size=IMG_SIZE, batch_size=BATCH)
        test_ds = test_ds.cache().prefetch(AUTOTUNE)

        model_path = os.path.join(DATA_DIR, "hurricane_model.h5")
        final_model_path = os.path.join(DATA_DIR, "final_hurricane_model.h5")
        model = load_model(model_path)

        loss, acc = model.evaluate(test_ds)
        print(f"Test Accuracy: {acc*100:.2f}%")

        os.rename(model_path, final_model_path)
        print(f"Final model saved at {final_model_path}")

        metrics_path = os.path.join(DATA_DIR, "metrics.txt")
        with open(metrics_path, "w") as f:
            f.write(f"loss: {loss}\naccuracy: {acc}\n")
        print(open(metrics_path).read())

    evaluate_task = PythonOperator(
        task_id="evaluate_and_save",
        python_callable=evaluate_and_save_model
    )

    # DAG dependencies
    fetch_s3_task >> prepare_task >> train_task >> evaluate_task
