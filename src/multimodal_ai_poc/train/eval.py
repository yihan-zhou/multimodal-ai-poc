from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import mlflow
import pandas as pd
import ray
from loguru import logger
from sklearn.metrics import multilabel_confusion_matrix

from multimodal_ai_poc.train.model import TorchPredictor, add_class


@dataclass
class BatchMetricOutput:
    """Dataclass to demonstrate output of batch_metric as originally implemented"""

    true_neg: list[float]
    false_neg: list[float]
    true_pos: list[float]
    false_pos: list[float]


def batch_metric(batch: dict[str, Any]) -> dict[str, list[float]]:
    """Calculate classification metrics on a batch of samples.

    See the Batch dataclass for the batch schema.
    """
    labels = batch["label"]
    preds = batch["prediction"]
    mcm = multilabel_confusion_matrix(labels, preds)
    tn, fp, fn, tp = [], [], [], []
    for i in range(mcm.shape[0]):
        tn.append(mcm[i, 0, 0])  # True negatives
        fp.append(mcm[i, 0, 1])  # False positives
        fn.append(mcm[i, 1, 0])  # False negatives
        tp.append(mcm[i, 1, 1])  # True positives

    # NOTE: typehinting functions like this that will be applied to ray.data.Datasets is a bit tricky
    # It would be more descriptive to return e.g. a dataclass but I don't know how to integrate that with ds.map_batches
    # For now, leaving this here in case it can be rectified later after closer investigation of map_batches
    # result = ConfusionMatrixOutput(
    #     true_neg=tn,
    #     false_neg=fn,
    #     true_pos=tp,
    #     false_pos=fp
    # )
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp}


def get_best_run(model_registry: Path, experiment_name: str) -> pd.Series:
    """Get the best run from an MLFlow experiment (on a hard-coded metric)."""
    # Sorted runs
    mlflow.set_tracking_uri(f"file:{str(model_registry)}")
    sorted_runs: pd.DataFrame = mlflow.search_runs(
        experiment_names=[experiment_name], order_by=["metrics.val_loss ASC"]
    )
    best_run = sorted_runs.iloc[0]
    return best_run


def eval_classifier() -> None:
    """Evaluate the best TorchPredictor experiment on the test set."""

    # From experiment logged in training run
    mlflow_model_registry = Path("/tmp/mlflow/doggos")  # nosec [B108:hardcoded_tmp_directory]
    mlflow_experiment_name = "doggos"
    best_run: pd.Series = get_best_run(
        model_registry=mlflow_model_registry, experiment_name=mlflow_experiment_name
    )
    artifacts_dir = Path(urlparse(best_run.artifact_uri).path)
    logger.info(f"{best_run=}")
    logger.info(f"{artifacts_dir=}")

    # Load and preproces eval dataset
    DEFAULT_N_TEST_LIMIT = 10
    predictor = TorchPredictor.from_artifacts_dir(artifacts_dir=artifacts_dir)
    test_ds = ray.data.read_images("s3://doggos-dataset/test", include_paths=True)
    test_ds = test_ds.map(add_class)
    test_ds = test_ds.limit(DEFAULT_N_TEST_LIMIT)
    test_ds = predictor.preprocessor.transform(ds=test_ds)

    pred_ds = test_ds.map_batches(
        predictor,
        concurrency=4,
        batch_size=64,
        # num_gpus=1,
        # accelerator_type="T4",
    )
    logger.debug(pred_ds.take(1))

    # Aggregated metrics after processing all batches
    metrics_ds = pred_ds.map_batches(batch_metric)
    aggregate_metrics = metrics_ds.sum(["TN", "FP", "FN", "TP"])

    # Aggregate the confusion matrix components across all batches
    tn = aggregate_metrics["sum(TN)"]
    fp = aggregate_metrics["sum(FP)"]
    fn = aggregate_metrics["sum(FN)"]
    tp = aggregate_metrics["sum(TP)"]

    # Calculate and log metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    logger.info(f"Precision: {precision:.2f}")
    logger.info(f"Recall: {recall:.2f}")
    logger.info(f"F1: {f1:.2f}")
    logger.info(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    eval_classifier()
