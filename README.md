# multimodal-ai-poc

[![check.yml](https://github.com/yihan-zhou/multimodal-ai-poc/actions/workflows/check.yml/badge.svg)](https://github.com/yihan-zhou/multimodal-ai-poc/actions/workflows/check.yml)
[![publish.yml](https://github.com/yihan-zhou/multimodal-ai-poc/actions/workflows/publish.yml/badge.svg)](https://github.com/yihan-zhou/multimodal-ai-poc/actions/workflows/publish.yml)
[![Documentation](https://img.shields.io/badge/documentation-available-brightgreen.svg)](https://yihan-zhou.github.io/multimodal-ai-poc/)
[![License](https://img.shields.io/github/license/yihan-zhou/multimodal-ai-poc)](https://github.com/yihan-zhou/multimodal-ai-poc/blob/main/LICENCE.txt)
[![Release](https://img.shields.io/github/v/release/yihan-zhou/multimodal-ai-poc)](https://github.com/yihan-zhou/multimodal-ai-poc/releases)

# Description	

A demo for multimodal AI POC, inspired by https://github.com/anyscale/multimodal-ai/blob/main/README.md

This demonstrates using Ray libraries to build a dog image classifier on top of a pre-trained embedding model.

Embeddings are generated using Ray Data for distributed processing. This utilizes a pre-trained `CLIPModel` loaded from HuggingFace.

The classifier is trained using a `TorchTrainer` from the Ray Train library. It is a simple 2-layer NN with FC1 > ReLU > FC2. 

This example uses the `s3://doggos-dataset` which appears custom to this example. It is split into train, val, test sets of sizes 2880, 720. Images are `shape=(500, 375, 3), dtype=uint8`.

```aiignore
import ray

train_ds = ray.data.read_images("s3://doggos-dataset/train", include_paths=True, shuffle="files")
val_ds = ray.data.read_images("s3://doggos-dataset/val", include_paths=True)
test_ds = ray.data.read_images("s3://doggos-dataset/test", include_paths=True)

train_ds.count()
>>> 2880

record = train_ds.take(1)
record[0]['image'].shape, record[0]['image'].dtype 
>>> (500, 375, 3), "uint8"
```

Some relevant docs:
- HuggingFace: [Transformers - CLIP models](https://huggingface.co/docs/transformers/en/model_doc/clip)

> CLIP is a is a multimodal vision and language model motivated by overcoming the fixed number of object categories when training a computer vision model. CLIP learns about images directly from raw text by jointly training on 400M (image, text) pairs. Pretraining on this scale enables zero-shot transfer to downstream tasks. CLIP uses an image encoder and text encoder to get visual features and text features. Both features are projected to a latent space with the same number of dimensions and their dot product gives a similarity score.

- Ray Docs: [Ray Data > User Guides > Transforming Data](https://docs.ray.io/en/latest/data/transforming-data.html#transforming-batches)
- - Describes transformations for `ray.data.Dataset`s as used in our embedding generation preprocessing
- Ray Docs: [Ray Train > PyTorch Guide: Get Started with Distributed Training using PyTorch](https://docs.ray.io/en/latest/train/getting-started-pytorch.html#train-pytorch)
- - Demonstrates the distributed training job config implemented here
- Ray Docs: [Ray Train API > ray.train.torch.TorchTrainer](https://docs.ray.io/en/latest/train/api/doc/ray.train.torch.TorchTrainer.html)

# Installation

Initialize your project with the provided `just` command.
```bash	
# Install dependencies and pre-commit hooks	
uv run just install	
```
# Usage

## Dog Image Classifier

Run the following script to train and evaluate the classifier.

Note the scripts don't yet include:
- Parametrization of training or eval settings. For example, the resources are hard-coded.
- Persistence of MLFlow artifacts (the model registry is configured to `/tmp` so it eventually gets cleaned up)

```aiignore
python multimodal_ai_poc/train/train.py
```

```aiignore
python multimodal_ai_poc/train/eval.py
```

### Embeddings

There is a module that uses a pre-trained CLIPModel to generate image embeddings. It can be run individually (this was done as a practice exercise) and is also used in the training pipeline.

Example usage:
```aiignore
python multimodal_ai_poc/embeddings.py \
    --dataset-uri s3://doggos-dataset/train \
    --num-images 100 \
    --model-id openai/clip-vit-base-patch32 \
    --num-ray-workers 4 \
    --embeddings-output-dir embeddings
```

### Similar Images

There is a module that returns the topn most similar images given a set of embeddings. (This was largely reproduced from [01-Batch-Inference.ipynb](https://github.com/anyscale/multimodal-ai/blob/main/notebooks/01-Batch-Inference.ipynb) as a practice exercise.)

Example usage:
```aiignore
python multimodal_ai_poc/similar_images.py \
    --image-url https://doggos-dataset.s3.us-west-2.amazonaws.com/samara.png \
    --embeddings-dir embeddings \
    --model-id openai/clip-vit-base-patch32 \
    --num-matches 10
```

Example for general Docker instructions:
```bash	
# Invoke docker compose	
uv run just docker-compose

# Or run with docker compose	
docker compose up --build

# Or run with docker	
# Note: specify platform if running on Apple M chip 	
docker build --platform linux/amd64 -t multimodal-ai-poc-image -f Dockerfile .	
docker run -it --platform linux/amd64 --name multimodal-ai-poc-ctr -p 8000:8000 multimodal-ai-poc-image	
```

## Development Features

Use the provided `just` commands to manage your development workflow:

- `uv run just check`: Run code quality, type, security, and test checks.
- `uv run just clean`: Clean up generated files.
- `uv run just commit`: Commit changes to your repository.
- `uv run just doc`: Generate API documentation.
- `uv run just docker`: Build and run your Docker image.
- `uv run just format`: Format your code with Ruff.
- `uv run just install`: Install dependencies, pre-commit hooks, and GitHub rulesets.
- `uv run just package`: Build your Python package.
- `uv run just project`: Run the project in the CLI.

* **Streamlined Project Structure:** A well-defined directory layout for source code, tests, documentation, tasks, and Docker configurations.
Uv Integration: Effortless dependency management and packaging with [uv](https://docs.astral.sh/uv/).
* **Automated Testing and Checks:** Pre-configured workflows using [Pytest](https://docs.pytest.org/), [Ruff](https://docs.astral.sh/ruff/), [Mypy](https://mypy.readthedocs.io/), [Bandit](https://bandit.readthedocs.io/), and [Coverage](https://coverage.readthedocs.io/) to ensure code quality, style, security, and type safety.
* **Pre-commit Hooks:** Automatic code formatting and linting with [Ruff](https://docs.astral.sh/ruff/) and other pre-commit hooks to maintain consistency.
* **Dockerized Deployment:** Dockerfile and docker-compose.yml for building and running the package within a containerized environment ([Docker](https://www.docker.com/)).
* **uv+just Task Automation:** [just](https://github.com/casey/just) commands to simplify development workflows such as cleaning, installing, formatting, checking, building, documenting and running the project.
* **Comprehensive Documentation:** [pdoc](https://pdoc.dev/) generates API documentation, and Markdown files provide clear usage instructions.
* **GitHub Workflow Integration:** Continuous integration and deployment workflows are set up using [GitHub Actions](https://github.com/features/actions), automating testing, checks, and publishing.
