import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from ray.train.torch import get_device

from multimodal_ai_poc.train.preprocessor import Preprocessor


def collate_fn(batch: dict[str, Any]) -> dict[str, Any]:
    """Ensure tensors have the proper data type."""
    dtypes = {"embedding": torch.float32, "label": torch.int64}
    tensor_batch = {}
    for key in dtypes.keys():
        if key in batch:
            tensor_batch[key] = torch.as_tensor(
                batch[key],
                dtype=dtypes[key],
                device=get_device(),
            )
    return tensor_batch


def add_class(row: dict[str, Any]) -> dict[str, Any]:
    row["class"] = row["path"].rsplit("/", 3)[-2]
    return row


class ClassificationModel(torch.nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout_p: float, num_classes: int):
        super().__init__()
        # Hyperparameters
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.num_classes = num_classes

        # Define layers
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        z = self.fc1(batch["embedding"])
        z = self.batch_norm(z)
        z = self.relu(z)
        z = self.dropout(z)
        z = self.fc2(z)
        return z

    @torch.inference_mode()
    def predict(self, batch: dict[str, Any]) -> npt.NDArray[np.uint8]:
        z = self(batch)
        y_pred = torch.argmax(z, dim=1).cpu().numpy()
        return y_pred

    @torch.inference_mode()
    def predict_probabilities(self, batch: dict[str, Any]) -> npt.NDArray[np.float32]:
        z = self(batch)
        y_probs = F.softmax(z, dim=1).cpu().numpy()
        return y_probs

    def save(self, dp: str) -> None:
        Path(dp).mkdir(parents=True, exist_ok=True)
        with open(Path(dp, "args.json"), "w") as fp:
            json.dump(
                {
                    "embedding_dim": self.embedding_dim,
                    "hidden_dim": self.hidden_dim,
                    "dropout_p": self.dropout_p,
                    "num_classes": self.num_classes,
                },
                fp,
                indent=4,
            )
        torch.save(self.state_dict(), Path(dp, "model.pt"))

    @classmethod
    def load(cls, args_fp: Path, state_dict_fp: Path, device: str="cpu") -> "ClassificationModel":
        with open(args_fp, "r") as fp:
            model = cls(**json.load(fp))
        model.load_state_dict(torch.load(state_dict_fp, map_location=device))  # nosec [B614:pytorch_load]
        logger.info(f"{type(model)=}")
        return model


class TorchPredictor:
    def __init__(self, preprocessor: Preprocessor, model: ClassificationModel):
        self.preprocessor = preprocessor
        self.model = model
        self.model.eval()

    def __call__(self, batch: dict[str, Any], device="cpu") -> dict[str, Any]:
        self.model.to(device)
        batch["prediction"] = self.model.predict(collate_fn(batch))
        return batch

    def predict_probabilities(self, batch: dict[str, Any], device: str="cpu") -> dict[str, Any]:
        self.model.to(device)
        predicted_probabilities = self.model.predict_probabilities(collate_fn(batch))
        batch["probabilities"] = [
            {
                self.preprocessor.label_to_class[i]: float(prob)
                for i, prob in enumerate(probabilities)
            }
            for probabilities in predicted_probabilities
        ]
        return batch

    @classmethod
    def from_artifacts_dir(cls, artifacts_dir: Path) -> "TorchPredictor":
        class_to_label_path = artifacts_dir / "class_to_label.json"
        with open(class_to_label_path, "r") as fp:
            class_to_label = json.load(fp)
        preprocessor = Preprocessor(class_to_label=class_to_label)
        model = ClassificationModel.load(
            args_fp=artifacts_dir / "args.json",
            state_dict_fp=artifacts_dir / "model.pt",
        )
        return cls(preprocessor=preprocessor, model=model)
