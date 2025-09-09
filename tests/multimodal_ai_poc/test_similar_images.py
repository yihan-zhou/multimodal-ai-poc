from pathlib import Path
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pytest
import ray
from pytest_mock import MockerFixture

from multimodal_ai_poc.embeddings import EmbedImages
from multimodal_ai_poc.similar_images import (
    SimilarImage,
    embed_image,
    get_top_matches,
    load_embeddings_ds,
)


@pytest.fixture
def image_array() -> npt.NDArray[np.uint8]:
    """Create a sample image array."""
    return np.ones((10, 10, 3), dtype=np.uint8) * 255


@pytest.fixture
def embedding_generator(mocker: MockerFixture) -> EmbedImages:
    """Create a mock embedding generator."""
    mock_generator = mocker.MagicMock()
    mock_generator.return_value = {"embedding": [np.ones(512, dtype=np.float32)]}
    return mock_generator


def test_embed_image(embedding_generator: EmbedImages, image_array: npt.NDArray[np.uint8]):
    """Test embed_image function."""
    with patch("multimodal_ai_poc.similar_images.url_to_array", return_value=image_array):
        result = embed_image("https://example.com/image.jpg", embedding_generator)

    embedding_generator.assert_called_once_with({"image": [image_array]})  # type: ignore[attr-defined]
    assert result.shape == (512,)
    assert result.dtype == np.float32


@patch("multimodal_ai_poc.similar_images.ray.data.read_parquet")
def test_load_embeddings_ds(mock_read_parquet, embeddings_ds: ray.data.Dataset):
    """Test load_embeddings_ds function."""
    mock_read_parquet.return_value = embeddings_ds
    embeddings_path = Path("/path/to/embeddings")

    result = load_embeddings_ds(embeddings_path)

    mock_read_parquet.assert_called_once_with(str(embeddings_path))
    assert result == embeddings_ds


def test_get_top_matches(embeddings_ds: ray.data.Dataset, mocker: MockerFixture):
    """Test get_top_matches function."""
    query_embedding = np.ones(512, dtype=np.float32)
    n = 2
    results = get_top_matches(query_embedding, embeddings_ds, n=n)

    assert len(results) == n
    assert isinstance(results[0], SimilarImage)

    # Test with n larger than available data
    results_large_n = get_top_matches(query_embedding, embeddings_ds, n=10)
    assert len(results_large_n) == embeddings_ds.count

    # Test with empty dataset
    empty_ds = mocker.MagicMock()
    empty_ds.take_all.return_value = []
    results_empty = get_top_matches(query_embedding, empty_ds)
    assert results_empty == []
