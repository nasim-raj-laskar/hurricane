"""Model evaluation tasks"""
import os
import numpy as np
import tensorflow as tf
import mlflow
from sklearn.metrics import precision_recall_fscore_support
from airflow.hooks.base import BaseHook  # type: ignore
from airflow.models import Variable  # type: ignore
from config import IMG_SIZE, BATCH, EPOCHS, LEARNING_RATE, OPTIMIZER


def _setup_mlflow():
    """Setup MLflow tracking"""
    mlflow_conn = BaseHook.get_connection('mlflow_dagshub')
    os.environ['MLFLOW_TRACKING_USERNAME'] = mlflow_conn.login
    os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_conn.password
    mlflow.set_tracking_uri(mlflow_conn.extra_dejson['tracking_uri'])


def _push_prometheus_metrics(push_url, run_id, acc, loss, y_true, y_pred):
    """Push metrics to Prometheus"""
    try:
        from utils.metrics import push_additional_metrics, start_continuous_mock_gpu_metrics, push_classification_metrics

        params = {
            "batch_size": BATCH,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "img_size": IMG_SIZE,
            "optimizer": OPTIMIZER
        }

        push_additional_metrics(push_url, run_id, acc, loss, params)
        push_classification_metrics(push_url, run_id, y_true, y_pred, ['no_damage', 'damage'])
        start_continuous_mock_gpu_metrics(push_url, run_id)
    except Exception as e:
        print("[WARN] Failed to push test metrics to Prometheus:", e)


def evaluate_model(**context):
    """Evaluate trained model on test set"""
    dirs = context['task_instance'].xcom_pull(task_ids='load_datasets')
    run_id = context['task_instance'].xcom_pull(task_ids='build_and_train', key='run_id')

    _setup_mlflow()

    model = tf.keras.models.load_model('/tmp/trained_model.h5')

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dirs["test"], image_size=IMG_SIZE, batch_size=BATCH
    ).cache().prefetch(tf.data.AUTOTUNE)

    loss, acc = model.evaluate(test_ds)

    # Get predictions
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_pred = (model.predict(test_ds) > 0.5).astype(int).flatten()

    # Log to MLflow
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("Test Accuracy", acc)
        mlflow.log_metric("Test Loss", loss)

        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        mlflow.log_metric("F1 Score", f1)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)

    # Push to Prometheus
    push_url = os.getenv('PUSHGATEWAY_URL') or Variable.get('PUSHGATEWAY_URL', default_var=None)
    if push_url:
        print(f"[METRICS] Using Pushgateway URL: {push_url}")
        _push_prometheus_metrics(push_url, run_id, acc, loss, y_true, y_pred)
    else:
        print("[METRICS] Skipped pushing additional metrics (no URL found).")

    return {'accuracy': acc, 'loss': loss}
