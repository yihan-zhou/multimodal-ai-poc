from unittest.mock import patch

import numpy as np
import ray
from transformers import CLIPModel, CLIPProcessor

from multimodal_ai_poc.embeddings import (
    EmbedImages,
    generate_embeddings,
    get_image_ds,
)


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
    mock_clip_processor_class,
    mock_clip_model_class,
    clip_processor: CLIPProcessor,
    clip_model: CLIPModel,
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
    clip_processor.assert_called_once()  # type: ignore[attr-defined]
    clip_model.get_image_features.assert_called_once()
    assert "embedding" in result


@patch("multimodal_ai_poc.embeddings.get_image_ds")
def test_generate_embeddings(
    mock_get_image_ds, image_ds: ray.data.Dataset, embeddings_ds: ray.data.Dataset
):
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
