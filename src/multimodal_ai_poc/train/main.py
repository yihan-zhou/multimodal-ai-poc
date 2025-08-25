import shutil
from pathlib import Path
from typing import Any

import ray
from loguru import logger

from multimodal_ai_poc.train.model import ClassificationModel
from multimodal_ai_poc.train.preprocessor import Preprocessor

DEFAULT_N_TRAIN_LIMIT = 80
DEFAULT_N_VAL_LIMIT = 20


# TODO: move somewhere more appropriate
def add_class(row: dict[str, Any]) -> dict[str, Any]:
    row["class"] = row["path"].rsplit("/", 3)[-2]
    return row


def preprocess(train_limit: int | None = None, val_limit: int | None = None) -> Preprocessor:
    # Load
    logger.debug("Loading data")
    train_ds = ray.data.read_images(
        "s3://doggos-dataset/train", include_paths=True, shuffle="files"
    )
    train_ds = train_ds.map(add_class)
    val_ds = ray.data.read_images("s3://doggos-dataset/val", include_paths=True)
    val_ds = val_ds.map(add_class)
    if train_limit:
        train_ds = train_ds.limit(train_limit)
    if val_limit:
        val_ds = val_ds.limit(val_limit)

    # Preprocess
    logger.debug("Preprocessing data")
    preprocessor = Preprocessor()
    preprocessor = preprocessor.fit(train_ds, column="class")
    train_ds = preprocessor.transform(ds=train_ds)
    val_ds = preprocessor.transform(ds=val_ds)

    # Write processed data
    preprocessed_data_path = Path(__file__).parent.parent / "preprocessed_data"
    if preprocessed_data_path.exists():
        logger.info(f"{preprocessed_data_path=} already exists, deleting")
        shutil.rmtree(preprocessed_data_path)
    preprocessed_train_path = preprocessed_data_path / "preprocessed_train"
    preprocessed_val_path = preprocessed_data_path / "preprocessed_val"
    train_ds.write_parquet(preprocessed_train_path)
    val_ds.write_parquet(preprocessed_val_path)

    logger.info(f"Wrote to {preprocessed_train_path=} {preprocessed_val_path=}")

    return preprocessor


def main() -> None:
    logger.info("Preprocessing")
    preprocessor: Preprocessor = preprocess(
        train_limit=DEFAULT_N_TRAIN_LIMIT, val_limit=DEFAULT_N_VAL_LIMIT
    )

    logger.info("Initializing model")
    num_classes = len(preprocessor.classes)
    model = ClassificationModel(
        embedding_dim=512,
        hidden_dim=256,
        dropout_p=0.3,
        num_classes=num_classes,
    )
    print(model)


if __name__ == "__main__":
    main()
