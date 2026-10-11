import time
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

LABEL_ORDER = ["NEG", "NEU", "POS"]

def synchronize_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "classification_report": classification_report(
            y_true, y_pred, labels=LABEL_ORDER, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABEL_ORDER),
    }

def timed_inference(pipeline_obj, texts, batch_size=32, max_length=128):
    warmup = texts[:min(batch_size, len(texts))]
    pipeline_obj(warmup, batch_size=batch_size, truncation=True, max_length=max_length)
    synchronize_cuda()
    start = time.perf_counter()
    outputs = pipeline_obj(
        texts,
        batch_size=batch_size,
        truncation=True,
        max_length=max_length,
    )
    synchronize_cuda()
    elapsed = time.perf_counter() - start
    return outputs, {
        "total_seconds": elapsed,
        "latency_ms_per_item": elapsed / len(texts) * 1000,
        "throughput_items_per_second": len(texts) / elapsed,
    }
