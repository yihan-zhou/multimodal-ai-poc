"""

Example usage: python multimodal_ai_poc/embeddings.py -d s3://doggos-dataset/train -n 100 -m openai/clip-vit-base-patch32 -w 4 -o embeddings
"""
import argparse
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


class EmbedImages:
    def __init__(self, model_id: str, device: str):
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id)
        self.model.to(device)
        self.device = device

    def __call__(self, batch):
        # Load and preprocess images
        images = [Image.fromarray(np.uint8(img)).convert("RGB") for img in batch["image"]]
        inputs: BatchEncoding = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)

        # Generate embeddings
        with torch.inference_mode():
            batch["embedding"] = self.model.get_image_features(**inputs).cpu().numpy()

        return batch


def _add_class(row: dict[str, Any]) -> dict[str, Any]:
    row["class"] = row["path"].rsplit("/", 3)[-2]
    return row


def get_image_ds(data_uri: str, limit: int | None = None) -> ray.data.Dataset:
    """Get an image Dataset from a cloud storage URI.

    Args:
        data_uri: Storage URI
        limit: Optional limit to apply to dataset
    Return:
        ray.data.Dataset
    """
    ds = ray.data.read_images(
        paths=data_uri,
        include_paths=True,
        shuffle="files",
    )
    ds = ds.map(_add_class, num_cpus=1, num_gpus=0, concurrency=4)
    if limit:
        ds = ds.limit(limit)
    return ds


def generate_embeddings():


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Return: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Compare production and staging model performance for smoke detection."
    )
    parser.add_argument("--dataset_uri", "-d", type=str, required=True, default="s3://doggos-dataset/train", help="URI to dataset, default=s3://doggos-dataset/train")
    parser.add_argument(
        "--num_images",
        "-n",
        type=int,
        required=False,
        default=100,
        help="Number of images to process from the dataset, default=100",
    )
    parser.add_argument(
        "--model_id",
        "-m",
        type=str,
        required=True,
        default="openai/clip-vit-base-patch32",
        help="CLIPModel to use for embedding generation, default=openai/clip-vit-base-patch32",
    )
    parser.add_argument(
        "--embeddings_output_dir",
        "-o",
        type=str,
        required=False,
        default=Path(__file__).parent / "embeddings",
        help="Directory to save embeddings, default=./embeddings",
    )
    parser.add_argument(
        "--num_ray_workers",
        "-w",
        type=int,
        required=False,
        default=4,
        help="Number of Ray Data workers used in emebdding generation, default=4",
    )

    return parser.parse_args()

def main():
    """Generate embeddings for a dataset given CLI arguments.

    Return:
        None
    """
    args = parse_args()
    output_dir = Path(__file__).parent / args.embeddings_output_dir
    if output_dir.exists():
        logger.info(f"{output_dir=} already exists, deleting")
        shutil.rmtree(output_dir)
    else:
        # Ensure directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Getting {args.num_images=} from dataset {args.dataset_uri=}")
    ds = get_image_ds(data_uri=args.dataset_uri, limit=args.num_images)

    logger.info(f"Generating embeddings with {args.model_id=}")
    embeddings_ds = ds.map_batches(
        EmbedImages,
        fn_constructor_kwargs={
            "model_id": args.model_id,
            "device": "cpu",
            # "device": "cuda",
        },  # class kwargs
        fn_kwargs={},  # __call__ kwargs
        concurrency=args.num_ray_workers,
        # batch_size=64,
        # num_gpus=1,
        # accelerator_type="L4",
    )
    embeddings_ds = embeddings_ds.drop_columns(["image"])  # remove image column

    logger.info("Saving embeddings")
    embeddings_ds.write_parquet(str(output_dir))

if __name__ == "__main__":
    main()

