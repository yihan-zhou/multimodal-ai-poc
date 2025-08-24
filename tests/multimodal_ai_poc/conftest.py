import numpy as np
import pytest
import ray
import torch
from pytest_mock import MockerFixture
from transformers import CLIPModel, CLIPProcessor


@pytest.fixture
def image_ds(mocker: MockerFixture) -> ray.data.Dataset:
    """Create a mock Ray dataset with test images."""
    # Create sample image data
    image_data = np.ones((10, 10, 3), dtype=np.uint8) * 255

    # Create mock dataset
    mock_ds = mocker.MagicMock()
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
def embeddings_ds(mocker: MockerFixture) -> ray.data.Dataset:
    """Create a mock Ray dataset with embeddings."""
    # Create sample embedding data
    embedding_data = np.ones((512,), dtype=np.float32)

    # Create mock dataset
    mock_ds = mocker.MagicMock()
    mock_ds.columns = ["embedding", "path", "class"]
    mock_ds.map.return_value = mock_ds
    mock_ds.limit.return_value = mock_ds
    mock_ds.drop_columns.return_value = mock_ds
    mock_ds.write_parquet.return_value = None

    # Mock data that would be returned by the dataset
    mock_data = [
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
    mock_ds.__iter__.return_value = mock_data
    mock_ds.take_all.return_value = mock_data
    mock_ds.count = len(mock_data)

    return mock_ds


@pytest.fixture
def clip_model(mocker: MockerFixture) -> CLIPModel:
    """Create a mock CLIP model."""
    mock_model = mocker.MagicMock()
    # Mock the get_image_features method to return fake embeddings
    mock_model.get_image_features.return_value = torch.ones((2, 512))
    return mock_model


@pytest.fixture
def clip_processor(mocker: MockerFixture) -> CLIPProcessor:
    """Create a mock CLIP processor."""
    mock_processor = mocker.MagicMock()

    # Create a mock BatchEncoding object
    mock_batch_encoding = mocker.MagicMock()
    mock_batch_encoding.to = mocker.MagicMock(return_value=mock_batch_encoding)
    mock_batch_encoding.__getitem__ = (
        lambda _, key: torch.ones((2, 3, 224, 224)) if key == "pixel_values" else None
    )

    # Set up the processor to return the mock BatchEncoding
    mock_processor.return_value = mock_batch_encoding

    return mock_processor
