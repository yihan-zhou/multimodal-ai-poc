"""Display top N similar images based on embedding similarity.
    url = "https://doggos-dataset.s3.us-west-2.amazonaws.com/samara.png"
    embeddings_dir = Path(__file__).parent / "embeddings"
    model_id = "openai/clip-vit-base-patch32"
    n = 10
Example usage:
python multimodal_ai_poc/similar_images.py \
    -i https://doggos-dataset.s3.us-west-2.amazonaws.com/samara.png \
    -e embeddings \
    -m openai/clip-vit-base-patch32 \
    -n 10
"""

import argparse
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import ray
import requests
from loguru import logger
from PIL import Image

from doggos.embed import display_top_matches, get_top_matches
from multimodal_ai_poc.embeddings import EmbedImages


def url_to_array(url: str) -> npt.NDArray[np.uint8]:
    arr = np.array(Image.open(BytesIO(requests.get(url).content)).convert("RGB"))
    logger.debug(f"{np.info(arr)}")  # type: ignore[func-returns-value]
    return arr


def embed_image(url: str, embedding_generator: EmbedImages) -> npt.NDArray[np.float32]:
    image = url_to_array(url=url)
    embedding = embedding_generator({"image": [image]})["embedding"][0]
    logger.debug(f"{np.info(embedding)}")  # type: ignore[func-returns-value]
    return embedding


def load_embeddings_ds(embeddings_path: Path) -> ray.data.Dataset:
    return ray.data.read_parquet(str(embeddings_path))


def display_topn_similar_matches(
    image_url: str, embeddings_dir: Path, model_id: str, n: int = 10
) -> None:
    """Display top N matches based on embedding similarity.

    Args:
        image_url: str, input image URL
        embeddings_dir: Path, directory containing embeddings
        model_id: str, Pretrained embedding model ID
        n: int, number of similar matches to display

    Return:
        None
    """
    embedding_generator = EmbedImages(model_id=model_id, device="cpu")
    embedding = embed_image(url=image_url, embedding_generator=embedding_generator)
    embeddings_ds = load_embeddings_ds(embeddings_path=embeddings_dir)

    top_matches: list[dict[str, Any]] = get_top_matches(
        query_embedding=embedding, embeddings_ds=embeddings_ds, n=n
    )
    display_top_matches(image_url, top_matches)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Arguments for displaying top N similar matches for an image"
    )
    parser.add_argument("--image-url", "-i", type=str, required=True, help="Link to input image")
    parser.add_argument(
        "--embeddings-dir",
        "-e",
        type=str,
        required=False,
        default=Path(__file__).parent / "embeddings",
        help="Embeddings directory, default=./embeddings",
    )
    parser.add_argument(
        "--num-matches",
        "-n",
        type=int,
        required=False,
        default=10,
        help="Number of similar images to display, default=10",
    )
    parser.add_argument(
        "--model-id",
        "-m",
        type=str,
        required=True,
        default="openai/clip-vit-base-patch32",
        help="CLIPModel to use for embedding generation, default=openai/clip-vit-base-patch32",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    embeddings_dir = Path(__file__).parent / args.embeddings_dir
    display_topn_similar_matches(
        image_url=args.image_url,
        embeddings_dir=embeddings_dir,
        model_id=args.model_id,
        n=args.num_matches,
    )


if __name__ == "__main__":
    main()
