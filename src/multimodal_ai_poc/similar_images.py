"""Display top N similar images based on embedding similarity.

Example usage:
python multimodal_ai_poc/similar_images.py \
    -i https://doggos-dataset.s3.us-west-2.amazonaws.com/samara.png \
    -e embeddings \
    -m openai/clip-vit-base-patch32 \
    -n 10
"""

import argparse
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import numpy.typing as npt
import ray
import requests
from loguru import logger
from matplotlib import pyplot as plt
from PIL import Image
from scipy.spatial.distance import cdist

from multimodal_ai_poc.embeddings import EmbedImages


@dataclass
class SimilarImage:
    class_label: str
    path: str
    similarity: float


def get_top_matches(
    query_embedding: npt.NDArray[np.float32],
    embeddings_ds: ray.data.Dataset,
    n: int = 4,
) -> list[SimilarImage]:
    """Get top N matches based for query embedding.

    Args:
        query_embedding: npt.NDArray[np.float32], query embedding
        embeddings_ds: ray.data.Dataset, source embeddings
        n: int, number of matches

    Return:
        list[SimilarImage] containing class_label, path, and similarity
    """
    rows = embeddings_ds.take_all()
    if not rows:
        return []

    # Vectorise
    embeddings = np.vstack([r["embedding"] for r in rows]).astype(np.float32)
    sims = 1 - cdist([query_embedding], embeddings, metric="cosine")[0]

    # Stable top N in NumPy
    k = min(n, sims.size)
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]

    # Package results
    return [
        SimilarImage(class_label=rows[i]["class"], path=rows[i]["path"], similarity=float(sims[i]))
        for i in idx
    ]


def display_top_matches(image_url: str, matches: list[SimilarImage]) -> None:
    """Display top similar matches.

    Args:
        image_url: str, input image URL
        matches: list[SimilarImage], List of top matches based on embedding similarity

    Return:
        None
    """
    fig, axes = plt.subplots(1, len(matches) + 1, figsize=(15, 5))

    # Display query image
    axes[0].imshow(url_to_array(url=image_url))
    axes[0].axis("off")
    axes[0].set_title("Query image")

    # Display matches
    for i, match in enumerate(matches):
        bucket = match.path.split("/")[0]
        key = "/".join(match.path.split("/")[1:])
        url = f"https://{bucket}.s3.us-west-2.amazonaws.com/{key}"
        image = url_to_array(url=url)
        axes[i + 1].imshow(image)
        axes[i + 1].axis("off")
        axes[i + 1].set_title(f"{match.class_label} ({match.similarity:.2f})")

    plt.tight_layout()
    plt.show()


def url_to_array(url: str) -> npt.NDArray[np.uint8]:
    arr = np.array(Image.open(BytesIO(requests.get(url, timeout=10).content)).convert("RGB"))
    logger.debug(f"{arr.shape=} {arr.dtype=}")
    return arr


def embed_image(url: str, embedding_generator: EmbedImages) -> npt.NDArray[np.float32]:
    image = url_to_array(url=url)
    embedding = embedding_generator({"image": [image]})["embedding"][0]
    logger.debug(f"{embedding.shape=} {embedding.dtype=}")
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

    top_matches: list[SimilarImage] = get_top_matches(
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
