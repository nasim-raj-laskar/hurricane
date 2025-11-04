"""Model building and training tasks"""
import os
import warnings
import mlflow
import tensorflow as tf
from tensorflow.keras import Sequential, Input  # type: ignore
from tensorflow.keras.layers import (Dense, Dropout, Conv2D, MaxPooling2D,  # type: ignore
                                     SeparableConv2D, GlobalAveragePooling2D, Rescaling)
from airflow.hooks.base import BaseHook  # type: ignore
from airflow.models import Variable  # type: ignore
from config import IMG_SIZE, BATCH, EPOCHS, LEARNING_RATE, OPTIMIZER, MLFLOW_EXPERIMENT_NAME

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")


def _setup_mlflow():
    """Setup MLflow tracking"""
    conn = BaseHook.get_connection("mlflow_dagshub")

    os.environ["MLFLOW_TRACKING_USERNAME"] = conn.login
    os.environ["MLFLOW_TRACKING_PASSWORD"] = conn.password
    os.environ["MLFLOW_TRACKING_URI"] = conn.extra_dejson.get(
        "tracking_uri", "https://dagshub.com/nasim-raj-laskar/hurricane.mlflow"
    )

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def _build_model():
    """Build CNN model"""
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
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def _get_callbacks(run_id):
    """Get training callbacks"""
    try:
        from utils.metrics import TrainingMetricsCallback
    except Exception as e:
        print(f"[METRICS] Could not import TrainingMetricsCallback: {e}")
        return []

    push_url = os.getenv('PUSHGATEWAY_URL') or Variable.get('PUSHGATEWAY_URL', default_var=None)
    
    if push_url:
        print(f"[METRICS] Using Pushgateway URL: {push_url}")
        return [TrainingMetricsCallback(pushgateway_url=push_url, job=f"hurricane_{run_id}", run_id=run_id)]
    
    return []


def build_and_train(**context):
    """Build and train the model"""
    _setup_mlflow()

    dirs = context['task_instance'].xcom_pull(task_ids='load_datasets')

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dirs["train"], image_size=IMG_SIZE, batch_size=BATCH
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dirs["val"], image_size=IMG_SIZE, batch_size=BATCH
    )

    model = _build_model()

    with mlflow.start_run(run_name="hurricane_training") as run:
        callbacks = _get_callbacks(run.info.run_id)
        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

        # Log parameters and metrics
        mlflow.log_params({
            "epochs": EPOCHS,
            "batch_size": BATCH,
            "img_size": IMG_SIZE,
            "optimizer": OPTIMIZER,
            "learning_rate": LEARNING_RATE
        })
        mlflow.log_metric("train_accuracy", history.history['accuracy'][-1] * 100)
        mlflow.log_metric("val_accuracy", history.history['val_accuracy'][-1] * 100)
        mlflow.log_metric("train_loss", history.history['loss'][-1] * 100)
        mlflow.log_metric("val_loss", history.history['val_loss'][-1] * 100)

        # Log metrics to console
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

        # Save model
        model.save('/tmp/trained_model.h5')
        mlflow.log_artifact('/tmp/trained_model.h5', artifact_path="model")

        context['task_instance'].xcom_push(key='run_id', value=run.info.run_id)

    return True
