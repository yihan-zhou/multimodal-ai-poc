import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import numpy.typing as npt
import ray
import torch
import torch.nn.functional as F
from loguru import logger
from ray.data import DataIterator
from ray.train import Result
from ray.train.torch import TorchTrainer
from torch.optim import Optimizer

from multimodal_ai_poc.train.model import ClassificationModel, add_class, collate_fn
from multimodal_ai_poc.train.preprocessor import Preprocessor

DEFAULT_N_TRAIN_LIMIT = 32 * 20
DEFAULT_N_VAL_LIMIT = 32 * 2


@dataclass
class Batch:
    """Dataclass to demonstrate structure of a `batch` object as originally included in model code"""

    path: list[str]
    _class: list[str]
    label: list[int]
    embedding: npt.NDArray[np.float32]  # shape (5, 512)


@dataclass
class TensorBatch:
    """Dataclass to demonstrate structure of a `tensor_batch` object as originally included in model code"""

    label: torch.Tensor  # dtype: torch.int64
    embedding: torch.Tensor  # dtype: torch.float32


def setup_model_registry() -> Path:
    model_registry = Path("/tmp/mlflow/doggos")  # nosec [B108:hardcoded_tmp_directory]
    if model_registry.is_dir():
        shutil.rmtree(model_registry)  # clean up
    model_registry.mkdir(parents=True, exist_ok=True)
    return model_registry


def preprocess(
    train_limit: int | None = None, val_limit: int | None = None
) -> tuple[Preprocessor, Path, Path]:
    """

    Return:
        Preprocessor, preprocessed_train_path, preprocessed_val_path
    """
    # Load
    logger.debug("Loading data")
    train_ds = ray.data.read_images(
        "s3://doggos-dataset/train", include_paths=True, shuffle="files"
    )
    train_ds = train_ds.map(add_class)
    val_ds = ray.data.read_images("s3://doggos-dataset/val", include_paths=True)
    val_ds = val_ds.map(add_class)
    if train_limit:
        train_ds = train_ds.limit(train_limit)
    if val_limit:
        val_ds = val_ds.limit(val_limit)

    # Preprocess
    logger.debug("Preprocessing data")
    preprocessor = Preprocessor()
    preprocessor = preprocessor.fit(train_ds, column="class")
    train_ds = preprocessor.transform(ds=train_ds)
    val_ds = preprocessor.transform(ds=val_ds)

    # Write processed data
    preprocessed_data_path = Path(__file__).parent.parent / "preprocessed_data"
    if preprocessed_data_path.exists():
        logger.info(f"{preprocessed_data_path=} already exists, deleting")
        shutil.rmtree(preprocessed_data_path)
    preprocessed_train_path = preprocessed_data_path / "preprocessed_train"
    preprocessed_val_path = preprocessed_data_path / "preprocessed_val"
    train_ds.write_parquet(preprocessed_train_path)
    val_ds.write_parquet(preprocessed_val_path)

    logger.info(f"Wrote to {preprocessed_train_path=} {preprocessed_val_path=}")

    return preprocessor, preprocessed_train_path, preprocessed_val_path


def train_epoch(
    ds: DataIterator,
    batch_size: int,
    model: torch.nn.Module,
    num_classes: int,
    loss_fn: torch.nn.Module,
    optimizer: Optimizer,
) -> float:
    """Run one training epoch."""
    model.train()
    loss = 0.0
    ds_generator = ds.iter_torch_batches(batch_size=batch_size, collate_fn=collate_fn)
    for i, batch in enumerate(ds_generator):
        optimizer.zero_grad()  # Reset gradients.
        z = model(batch)  # Forward pass.
        targets = F.one_hot(batch["label"], num_classes=num_classes).float()
        J = loss_fn(z, targets)  # Define loss.
        J.backward()  # Backward pass.
        optimizer.step()  # Update weights.
        loss += (J.detach().item() - loss) / (i + 1)  # Cumulative loss
    return loss


def eval_epoch(
    ds: DataIterator,
    batch_size: int,
    model: torch.nn.Module,
    num_classes: int,
    loss_fn: torch.nn.Module,
) -> tuple[float, npt.NDArray, npt.NDArray]:
    """Run one evaluation epoch."""
    model.eval()
    loss = 0.0
    y_trues: list[int] = []
    y_preds: list[int] = []
    ds_generator = ds.iter_torch_batches(batch_size=batch_size, collate_fn=collate_fn)
    with torch.inference_mode():
        for i, batch in enumerate(ds_generator):
            z = model(batch)
            targets = F.one_hot(
                batch["label"], num_classes=num_classes
            ).float()  # one-hot (for loss_fn)
            J = loss_fn(z, targets).item()
            loss += (J - loss) / (i + 1)
            y_trues.extend(batch["label"].cpu().numpy())
            y_preds.extend(torch.argmax(z, dim=1).cpu().numpy())
    return loss, np.vstack(y_trues), np.vstack(y_preds)


def train_loop_per_worker(config: dict[str, Any]) -> None:
    """Train loop input for TorchTrainer."""
    # NOTE: cannot use logger in this fn OR any fns invoked since it result in a serialization issue!
    # Hyperparameters.
    model_registry = config["model_registry"]
    experiment_name = config["experiment_name"]
    embedding_dim = config["embedding_dim"]
    hidden_dim = config["hidden_dim"]
    dropout_p = config["dropout_p"]
    lr = config["lr"]
    lr_factor = config["lr_factor"]
    lr_patience = config["lr_patience"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    num_classes = config["num_classes"]

    # Experiment tracking.
    if ray.train.get_context().get_world_rank() == 0:
        mlflow.set_tracking_uri(f"file:{model_registry}")
        mlflow.set_experiment(experiment_name)
        mlflow.start_run()
        mlflow.log_params(config)

    # Datasets.
    train_ds = ray.train.get_dataset_shard("train")
    val_ds = ray.train.get_dataset_shard("val")

    # Model.
    model = ClassificationModel(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout_p=dropout_p,
        num_classes=num_classes,
    )
    model = ray.train.torch.prepare_model(model)

    # Training components.
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_factor,
        patience=lr_patience,
    )

    # Training.
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        # Steps
        train_loss = train_epoch(train_ds, batch_size, model, num_classes, loss_fn, optimizer)
        val_loss, _, _ = eval_epoch(val_ds, batch_size, model, num_classes, loss_fn)
        scheduler.step(val_loss)

        # Checkpoint (metrics, preprocessor and model artifacts).
        with tempfile.TemporaryDirectory() as dp:
            model.module.save(dp=dp)
            metrics = dict(
                lr=optimizer.param_groups[0]["lr"], train_loss=train_loss, val_loss=val_loss
            )
            checkpoint_file = Path(dp) / "class_to_label.json"
            with open(checkpoint_file, "w") as fp:
                json.dump(config["class_to_label"], fp, indent=4)
            if ray.train.get_context().get_world_rank() == 0:  # only on main worker 0
                mlflow.log_metrics(metrics, step=epoch)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    mlflow.log_artifacts(dp)

    # End experiment tracking.
    if ray.train.get_context().get_world_rank() == 0:
        mlflow.end_run()


def train_classifier() -> Result:
    logger.info("Preprocessing")
    preprocessor, preprocessed_train_path, preprocessed_val_path = preprocess(
        train_limit=DEFAULT_N_TRAIN_LIMIT, val_limit=DEFAULT_N_VAL_LIMIT
    )

    logger.info("Initializing model")
    num_classes = len(preprocessor.classes)
    model = ClassificationModel(
        embedding_dim=512,
        hidden_dim=256,
        dropout_p=0.3,
        num_classes=num_classes,
    )
    logger.info(f"{model=}")

    logger.info("Setting up model registry")
    model_registry = setup_model_registry()

    # Train loop config.
    experiment_name = "doggos"
    train_loop_config = {
        "model_registry": model_registry,
        "experiment_name": experiment_name,
        "embedding_dim": 512,
        "hidden_dim": 256,
        "dropout_p": 0.3,
        "lr": 1e-3,
        "lr_factor": 0.8,
        "lr_patience": 3,
        "num_epochs": 2,
        "batch_size": 32,
    }
    # Scaling config
    num_workers = 8
    scaling_config = ray.train.ScalingConfig(
        num_workers=num_workers,
        resources_per_worker={"CPU": 1},
        # use_gpu=True,
        # resources_per_worker={"CPU": 8, "GPU": 2},
        # accelerator_type="T4",
    )

    # Load preprocessed datasets.
    preprocessed_train_ds = ray.data.read_parquet(str(preprocessed_train_path))
    preprocessed_val_ds = ray.data.read_parquet(str(preprocessed_val_path))

    # Trainer.
    train_loop_config["class_to_label"] = preprocessor.class_to_label
    train_loop_config["num_classes"] = len(preprocessor.class_to_label)
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        datasets={"train": preprocessed_train_ds, "val": preprocessed_val_ds},
    )
    # Train.
    results: Result = trainer.fit()
    logger.info(results)
    return results


if __name__ == "__main__":
    train_classifier()
