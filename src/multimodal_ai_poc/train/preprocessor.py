import json
from typing import Any

import numpy as np
import ray
import torch
from loguru import logger
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from transformers.tokenization_utils_base import BatchEncoding

# Need to pin revisions per CWE B615 during bandit checks - see model_refs_helper
DEFAULT_CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


# TODO: get from embeddings.py when changes merged available
class EmbedImages:
    def __init__(
        self, model_id: str, device: str = "cpu", model_revision: str = DEFAULT_CLIP_REVISION
    ):
        # Adding nosecs bc bandit only recognizes when the str literal is passed as revision
        self.processor = CLIPProcessor.from_pretrained(model_id, revision=model_revision)  # nosec B615
        self.model = CLIPModel.from_pretrained(model_id, revision=model_revision)  # nosec B615
        self.model.to(device)
        self.device = device

    def __call__(self, batch):
        # Load and preprocess images
        images = [Image.fromarray(np.uint8(img)).convert("RGB") for img in batch["image"]]
        inputs: BatchEncoding = self.processor(images=images, return_tensors="pt", padding=True).to(
            self.device
        )

        # Generate embeddings
        with torch.inference_mode():
            batch["embedding"] = self.model.get_image_features(**inputs).cpu().numpy()

        return batch


def convert_to_label(row: dict[str, Any], class_to_label: dict[str, str]) -> dict[str, Any]:
    if "class" in row:
        try:
            row["label"] = class_to_label[row["class"]]
        except KeyError:
            logger.warning(f"{row['class']=} not in {class_to_label=}")
            row["label"] = "UNK"
    return row


class Preprocessor:
    """Preprocessor class."""

    def __init__(self, class_to_label: dict[str, int] | None = None):
        # mutable defaults
        self.classes: list[str] = []
        self.class_to_label = class_to_label or {}
        self.label_to_class = {v: k for k, v in self.class_to_label.items()}

    def fit(self, ds: ray.data.Dataset, column: str) -> "Preprocessor":
        self.classes = ds.unique(column=column)
        self.class_to_label = {tag: i for i, tag in enumerate(self.classes)}
        self.label_to_class = {v: k for k, v in self.class_to_label.items()}
        return self

    def transform(
        self, ds: ray.data.Dataset, concurrency: int = 4, batch_size: int = 64, num_gpus: int = 1
    ):
        ds = ds.map(
            convert_to_label,  # type: ignore[]
            fn_kwargs={"class_to_label": self.class_to_label},
        )
        ds = ds.map_batches(
            EmbedImages,
            fn_constructor_kwargs={
                "model_id": "openai/clip-vit-base-patch32",
                "device": "cpu",
            },
            concurrency=concurrency,
            batch_size=batch_size,
            # num_gpus=num_gpus,
            # accelerator_type="T4",
        )
        ds = ds.drop_columns(["image"])
        return ds

    def save(self, fp):
        with open(fp, "w") as f:
            json.dump(self.class_to_label, f)
