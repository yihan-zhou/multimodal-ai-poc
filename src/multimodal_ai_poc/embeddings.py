import shutil
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from loguru import logger
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from transformers.tokenization_utils_base import BatchEncoding

MAX_IMAGES = 100


class EmbedImages:
    def __init__(self, model_id: str, device: str):
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id)
        self.model.to(device)
        self.device = device

    def __call__(self, batch):
        # Load and preprocess images
        logger.info(f"{type(batch)=}")
        images = [Image.fromarray(np.uint8(img)).convert("RGB") for img in batch["image"]]
        inputs: BatchEncoding = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)

        # Generate embeddings
        with torch.inference_mode():
            batch["embedding"] = self.model.get_image_features(**inputs).cpu().numpy()

        return batch


def _add_class(row: dict[str, Any]) -> dict[str, Any]:
    row["class"] = row["path"].rsplit("/", 3)[-2]
    return row


def get_image_ds(data_path: str, limit: int | None = None) -> ray.data.Dataset:
    ds = ray.data.read_images(
        paths=data_path,
        include_paths=True,
        shuffle="files",
    )
    ds = ds.map(_add_class, num_cpus=1, num_gpus=0, concurrency=4)
    if limit:
        ds = ds.limit(limit)
    return ds


if __name__ == "__main__":
    # Generate batch embeddings
    logger.info("Getting dataset")
    data_path = "s3://doggos-dataset/train"
    ds = get_image_ds(data_path, limit=MAX_IMAGES)

    logger.info("Generating embeddings")
    embeddings_ds = ds.map_batches(
        EmbedImages,
        fn_constructor_kwargs={
            "model_id": "openai/clip-vit-base-patch32",
            "device": "cpu",
            # "device": "cuda",
        },  # class kwargs
        fn_kwargs={},  # __call__ kwargs
        concurrency=4,
        # batch_size=64,
        # num_gpus=1,
        # accelerator_type="L4",
    )
    embeddings_ds = embeddings_ds.drop_columns(["image"])  # remove image column

    # Save to artifact storage.
    logger.info("Saving embeddings")
    embeddings_path = Path(__file__).parent / "embeddings"
    embeddings_path.mkdir(parents=True, exist_ok=True)  # ensure directory exists
    if embeddings_path.exists():
        shutil.rmtree(embeddings_path)  # clean up
    embeddings_ds.write_parquet(str(embeddings_path))
