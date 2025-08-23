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


def url_to_array(url: str) -> npt.NDArray:
    arr = np.array(Image.open(BytesIO(requests.get(url).content)).convert("RGB"))
    logger.info(f"{arr.dtype=} {np.info(arr)}")
    return arr

def embed_image(url: str, embedding_generator: EmbedImages) -> npt.NDArray:
    image = url_to_array(url=url)
    embedding = embedding_generator({"image": [image]})["embedding"][0]
    logger.info(f"{np.shape(embedding)=} {embedding.dtype=} {np.info(embedding)=}")
    return embedding

def load_embeddings_ds(embeddings_path: Path) -> ray.data.Dataset:
    return ray.data.read_parquet(str(embeddings_path))

def main():
    url = "https://doggos-dataset.s3.us-west-2.amazonaws.com/samara.png"
    embeddings_path = Path(__file__).parent / "embeddings"
    embedding_generator = EmbedImages(model_id="openai/clip-vit-base-patch32", device="cpu")
    embedding = embed_image(url=url, embedding_generator=embedding_generator)
    embeddings_ds = load_embeddings_ds(embeddings_path=embeddings_path)

    top_matches: list[dict[str, Any]] = get_top_matches(query_embedding=embedding, embeddings_ds=embeddings_ds, n=5)
    display_top_matches(url, top_matches)

if __name__=="__main__":
    main()
