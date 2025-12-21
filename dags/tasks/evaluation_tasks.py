"""Model evaluation tasks"""

import os
from airflow.hooks.base import BaseHook  # type: ignore
from airflow.models import Variable  # type: ignore
from dags.config import IMG_SIZE, BATCH, EPOCHS, LEARNING_RATE, OPTIMIZER


def _setup_mlflow():
    """Setup MLflow tracking"""
    import mlflow

    mlflow_conn = BaseHook.get_connection("mlflow_dagshub")
    os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_conn.login
    os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_conn.password
    mlflow.set_tracking_uri(mlflow_conn.extra_dejson["tracking_uri"])


def _push_prometheus_metrics(push_url, run_id, acc, loss, y_true, y_pred):
    """Push metrics to Prometheus"""
    try:
        from dags.utils.metrics import (
            push_additional_metrics,
            push_mock_gpu_metrics,
            push_classification_metrics,
        )

        params = {
            "batch_size": BATCH,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "img_size": IMG_SIZE,
            "optimizer": OPTIMIZER,
        }

        push_additional_metrics(push_url, run_id, acc, loss, params)
        push_classification_metrics(
            push_url, run_id, y_true, y_pred, ["no_damage", "damage"]
        )
        push_mock_gpu_metrics(push_url, run_id)

    except Exception as e:
        import traceback

        print("[WARN] Failed to push test metrics to Prometheus:", e)
        traceback.print_exc()


def evaluate_model(**context):
    """Evaluate trained model on test set"""

    # ✅ Heavy imports ONLY inside task
    import numpy as np
    import tensorflow as tf
    import mlflow
    from sklearn.metrics import precision_recall_fscore_support

    ti = context["task_instance"]

    dirs = ti.xcom_pull(task_ids="load_datasets")
    run_id = ti.xcom_pull(task_ids="build_and_train", key="run_id")

    _setup_mlflow()

    model = tf.keras.models.load_model("/tmp/trained_model.h5")

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dirs["test"], image_size=IMG_SIZE, batch_size=BATCH
    ).cache().prefetch(tf.data.AUTOTUNE)

    loss, acc = model.evaluate(test_ds)

    # Predictions
    y_true = np.concatenate([y for _, y in test_ds], axis=0)
    y_pred = (model.predict(test_ds) > 0.5).astype(int).flatten()

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_loss", loss)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary"
        )

        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

    # Prometheus
    push_url = os.getenv("PUSHGATEWAY_URL") or Variable.get(
        "PUSHGATEWAY_URL", default_var=None
    )

    if push_url:
        print(f"[METRICS] Using Pushgateway URL: {push_url}")
        _push_prometheus_metrics(push_url, run_id, acc, loss, y_true, y_pred)
    else:
        print("[METRICS] Skipped pushing additional metrics (no URL found).")

    return {"accuracy": acc, "loss": loss}
