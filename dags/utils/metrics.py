"""Prometheus helper for training metrics.

Provides a Keras callback that pushes per-epoch metrics to a Prometheus Pushgateway.

Usage:
  from utils.metrics import TrainingMetricsCallback

  cb = TrainingMetricsCallback(pushgateway_url=os.getenv('PUSHGATEWAY_URL'))
  model.fit(..., callbacks=[cb])

If PUSHGATEWAY_URL is not provided, the callback will only update an internal registry
and will not attempt to push (useful for local testing).
"""
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
import tensorflow as tf
from typing import Optional
import random 

class TrainingMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, pushgateway_url: Optional[str] = None, job: str = "hurricane_training", run_id: Optional[str] = None):
        super().__init__()
        self.pushgateway_url = pushgateway_url
        self.job = job
        self.run_id = run_id or "unknown"
        self.registry = CollectorRegistry()

        # Diagnostic print so Airflow task logs show whether callback was constructed
        print(f"[METRICS] TrainingMetricsCallback initialized. pushgateway_url={self.pushgateway_url}, job={self.job}")

        # Gauges labelled by epoch and run_id so you can query/plot each epoch/run as a timeseries
        self.g_train_acc = Gauge('train_accuracy_percent', 'Training accuracy (percent)', ['epoch', 'run_id'], registry=self.registry)
        self.g_val_acc = Gauge('val_accuracy_percent', 'Validation accuracy (percent)', ['epoch', 'run_id'], registry=self.registry)
        self.g_train_loss = Gauge('train_loss', 'Training loss', ['epoch', 'run_id'], registry=self.registry)
        self.g_val_loss = Gauge('val_loss', 'Validation loss', ['epoch', 'run_id'], registry=self.registry)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        e = str(epoch + 1)

        # Logs may contain accuracy as 0-1; convert accuracy to percent for easier display
        acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')

        if acc is not None:
            self.g_train_acc.labels(epoch=e, run_id=self.run_id).set(float(acc) * 100)
        if val_acc is not None:
            self.g_val_acc.labels(epoch=e, run_id=self.run_id).set(float(val_acc) * 100)
        if loss is not None:
            self.g_train_loss.labels(epoch=e, run_id=self.run_id).set(float(loss))
        if val_loss is not None:
            self.g_val_loss.labels(epoch=e, run_id=self.run_id).set(float(val_loss))




        if self.pushgateway_url:
            try:
                print(f"[METRICS] Pushing metrics to {self.pushgateway_url} for job={self.job}, run_id={self.run_id}, epoch={e}")
                push_to_gateway(self.pushgateway_url, job=self.job, registry=self.registry)
                print(f"[METRICS] Push successful for job={self.job}, run_id={self.run_id}, epoch={e}")
            except Exception:
                # Push should never crash training - swallow and continue
                import traceback
                print(f"[WARN] failed to push metrics to {self.pushgateway_url}")
                traceback.print_exc()




def push_classification_metrics(pushgateway_url: str, run_id: str, y_true, y_pred, class_names: list):
    """
    Push F1, precision, recall, and confusion matrix to Prometheus.
    """
    if not pushgateway_url:
        return
    
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    import numpy as np
    
    registry = CollectorRegistry()
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    cm = confusion_matrix(y_true, y_pred)
    
    g_f1 = Gauge('f1_score', 'F1 Score', ['run_id'], registry=registry)
    g_precision = Gauge('precision', 'Precision', ['run_id'], registry=registry)
    g_recall = Gauge('recall', 'Recall', ['run_id'], registry=registry)
    g_cm = Gauge('confusion_matrix', 'Confusion Matrix', ['run_id', 'true_label', 'pred_label'], registry=registry)
    
    g_f1.labels(run_id=run_id).set(f1)
    g_precision.labels(run_id=run_id).set(precision)
    g_recall.labels(run_id=run_id).set(recall)
    
    for i, true_label in enumerate(class_names):
        for j, pred_label in enumerate(class_names):
            g_cm.labels(run_id=run_id, true_label=true_label, pred_label=pred_label).set(int(cm[i][j]))
    
    try:
        push_to_gateway(pushgateway_url, job=f"classification_{run_id}", registry=registry)
        print(f"[METRICS] Pushed F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
    except Exception as e:
        print(f"[WARN] Failed to push classification metrics: {e}")

#Addition metics
def push_additional_metrics(pushgateway_url: str, run_id: str, test_acc: float, test_loss: float, params: dict):
    """
    Push test accuracy/loss and hyperparameters to Prometheus Pushgateway.
    """
    if not pushgateway_url:
        print("[METRICS] No Pushgateway URL provided. Skipping push.")
        return

    print(f"[METRICS] Pushing test metrics & hyperparameters for run_id={run_id}")
    registry = CollectorRegistry()

    # Test metrics
    g_test_acc = Gauge('test_accuracy_percent', 'Test accuracy (percent)', ['run_id'], registry=registry)
    g_test_loss = Gauge('test_loss', 'Test loss', ['run_id'], registry=registry)
    g_test_acc.labels(run_id=run_id).set(test_acc * 100)
    g_test_loss.labels(run_id=run_id).set(test_loss)

    # Hyperparams
    g_batch = Gauge('batch_size', 'Batch size used for training', ['run_id'], registry=registry)
    g_lr = Gauge('learning_rate', 'Learning rate used for training', ['run_id'], registry=registry)
    g_epochs = Gauge('epochs', 'Number of epochs trained', ['run_id'], registry=registry)
    g_img_w = Gauge('img_size_width', 'Input image width', ['run_id'], registry=registry)
    g_img_h = Gauge('img_size_height', 'Input image height', ['run_id'], registry=registry)
    g_optimizer = Gauge('optimizer_id', 'Optimizer identifier (1=Adam,2=SGD,3=RMSProp,... )', ['run_id'], registry=registry)

    g_batch.labels(run_id=run_id).set(params.get('batch_size', 0))
    g_lr.labels(run_id=run_id).set(params.get('learning_rate', 0))
    g_epochs.labels(run_id=run_id).set(params.get('epochs', 0))
    g_img_w.labels(run_id=run_id).set(params.get('img_size', (0, 0))[0])
    g_img_h.labels(run_id=run_id).set(params.get('img_size', (0, 0))[1])

    opt_map = {'adam': 1, 'sgd': 2, 'rmsprop': 3}
    g_optimizer.labels(run_id=run_id).set(opt_map.get(params.get('optimizer', '').lower(), 0))

    try:
        push_to_gateway(pushgateway_url, job=f"hurricane_{run_id}_summary", registry=registry)
        print("[METRICS]  Successfully pushed test metrics + hyperparams.")
    except Exception as e:
        import traceback
        print(f"[WARN] Failed to push additional metrics: {e}")
        traceback.print_exc()

## MOCK GPU metrics, as i dont have a gpu in my system :(
def push_mock_gpu_metrics(pushgateway_url: str, run_id: str):
    if not pushgateway_url:
        print("[GPU-MOCK] No pushgateway URL provided, skipping GPU metrics")
        return
    
    import random
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

    registry = CollectorRegistry()
    g_gpu = Gauge('gpu_utilization_percent', 'Mock GPU utilization (%)', ['run_id'], registry=registry)
    g_mem = Gauge('gpu_memory_used_mb', 'Mock GPU memory used (MB)', ['run_id'], registry=registry)
    g_temp = Gauge('gpu_temperature_celsius', 'Mock GPU temperature (°C)', ['run_id'], registry=registry)
    
    gpu = random.uniform(60, 85)
    mem = random.uniform(2500, 4500)
    temp = random.uniform(55, 75)

    g_gpu.labels(run_id=run_id).set(gpu)
    g_mem.labels(run_id=run_id).set(mem)
    g_temp.labels(run_id=run_id).set(temp)

    try:
        push_to_gateway(pushgateway_url, job=f"mock_gpu_{run_id}", registry=registry)
        print(f"[GPU-MOCK] Pushed GPU={gpu:.1f}%, MEM={mem:.0f}MB, TEMP={temp:.1f}°C")
    except Exception as e:
        import traceback
        print(f"[WARN] GPU mock push failed: {e}")
        traceback.print_exc()


