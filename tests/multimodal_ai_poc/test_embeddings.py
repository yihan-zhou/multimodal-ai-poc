from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import ray
import torch
from transformers import CLIPModel, CLIPProcessor

from multimodal_ai_poc.embeddings import (
    EmbedImages,
    generate_embeddings,
    get_image_ds,
)


@pytest.fixture
def image_ds() -> ray.data.Dataset:
    """Create a mock Ray dataset with test images."""
    # Create sample image data
    image_data = np.ones((10, 10, 3), dtype=np.uint8) * 255

    # Create mock dataset
    mock_ds = MagicMock()
    mock_ds.map.return_value = mock_ds
    mock_ds.limit.return_value = mock_ds
    mock_ds.drop_columns.return_value = mock_ds

    # Mock data that would be returned by the dataset
    mock_ds.__iter__.return_value = iter(
        [
            {"image": image_data, "path": "s3://test-bucket/class1/img1.jpg"},
            {"image": image_data, "path": "s3://test-bucket/class2/img2.jpg"},
        ]
    )

    return mock_ds

@pytest.fixture
def embeddings_ds() -> ray.data.Dataset:
    """Create a mock Ray dataset with embeddings."""
    # Create sample embedding data
    embedding_data = np.ones((512,), dtype=np.float32)

    # Create mock dataset
    mock_ds = MagicMock()
    mock_ds.columns = ["embedding", "path", "class"]
    mock_ds.map.return_value = mock_ds
    mock_ds.limit.return_value = mock_ds
    mock_ds.drop_columns.return_value = mock_ds
    mock_ds.write_parquet.return_value = None

    # Mock data that would be returned by the dataset
    mock_ds.__iter__.return_value = iter(
        [
            {
                "embedding": embedding_data,
                "path": "s3://test-bucket/class1/img1.jpg",
                "class": "class1",
            },
            {
                "embedding": embedding_data,
                "path": "s3://test-bucket/class2/img2.jpg",
                "class": "class2",
            },
        ]
    )

    return mock_ds

@pytest.fixture
def clip_model() -> CLIPModel:
    """Create a mock CLIP model."""
    mock_model = MagicMock()
    # Mock the get_image_features method to return fake embeddings
    mock_model.get_image_features.return_value = torch.ones((2, 512))
    return mock_model


@pytest.fixture
def clip_processor() -> CLIPProcessor:
    """Create a mock CLIP processor."""
    mock_processor = MagicMock()

    # Create a mock BatchEncoding object
    mock_batch_encoding = MagicMock()
    mock_batch_encoding.to = MagicMock(return_value=mock_batch_encoding)
    mock_batch_encoding.__getitem__ = (
        lambda _, key: torch.ones((2, 3, 224, 224)) if key == "pixel_values" else None
    )

    # Set up the processor to return the mock BatchEncoding
    mock_processor.return_value = mock_batch_encoding

    return mock_processor



@patch("multimodal_ai_poc.embeddings.ray.data.read_images")
def test_get_image_ds(mock_read_images, image_ds: ray.data.Dataset):
    """Test the get_image_ds function."""
    mock_read_images.return_value = image_ds

    # Test without limit
    _ = get_image_ds("s3://test-bucket")
    mock_read_images.assert_called_once()
    image_ds.map.assert_called_once()
    image_ds.limit.assert_not_called()

    # Test with limit
    mock_read_images.reset_mock()
    image_ds.map.reset_mock()
    mock_read_images.return_value = image_ds

    _ = get_image_ds("s3://test-bucket", limit=10)
    mock_read_images.assert_called_once()
    image_ds.map.assert_called_once()
    image_ds.limit.assert_called_once_with(10)


@patch("multimodal_ai_poc.embeddings.CLIPModel")
@patch("multimodal_ai_poc.embeddings.CLIPProcessor")
def test_embed_images(
    mock_clip_processor_class, mock_clip_model_class, clip_processor: CLIPProcessor, clip_model: CLIPModel
):
    """Test the EmbedImages class."""
    mock_clip_processor_class.from_pretrained.return_value = clip_processor
    mock_clip_model_class.from_pretrained.return_value = clip_model

    # Create test image data
    image_data = np.ones((10, 10, 3), dtype=np.uint8) * 255
    batch = {"image": [image_data, image_data]}

    # Initialize and call the embedder
    embedder = EmbedImages(model_id="test/model")
    result = embedder(batch)

    # Verify the model was called correctly
    clip_processor.assert_called_once()
    clip_model.get_image_features.assert_called_once()
    assert "embedding" in result


@patch("multimodal_ai_poc.embeddings.get_image_ds")
def test_generate_embeddings(mock_get_image_ds, image_ds: ray.data.Dataset, embeddings_ds: ray.data.Dataset):
    """Test the generate_embeddings function."""
    mock_get_image_ds.return_value = image_ds
    image_ds.map_batches.return_value = embeddings_ds

    # Call the function
    embeddings_ds = generate_embeddings(
        dataset_uri="s3://test-bucket", num_images=10, model_id="test/model", num_ray_workers=2
    )

    # Verify correct calls
    mock_get_image_ds.assert_called_once_with(data_uri="s3://test-bucket", limit=10)
    image_ds.map_batches.assert_called_once()
    assert "embedding" in embeddings_ds.columns
