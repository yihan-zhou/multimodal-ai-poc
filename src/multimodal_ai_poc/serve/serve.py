import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI
from loguru import logger
from PIL import Image
from ray import serve
from starlette.requests import Request
from transformers import CLIPModel, CLIPProcessor

from multimodal_ai_poc.similar_images import url_to_array
from multimodal_ai_poc.train.eval import get_artifacts_dir_best_run
from multimodal_ai_poc.train.model import TorchPredictor, collate_fn
from multimodal_ai_poc.train.preprocessor import DEFAULT_CLIP_REVISION

api = FastAPI(
    title="Dog Classifier",
)


def load_image_from_url(url: str) -> Image.Image:
    return Image.fromarray(np.uint8(url_to_array(url=url))).convert("RGB")


@serve.deployment
class ClassifierModel:
    def __init__(
        self,
        model_id: str,
        artifacts_dir: Path,
        device: str = "cpu",
        model_revision: str = DEFAULT_CLIP_REVISION,
    ):
        # Adding nosecs bc bandit only recognizes when the str literal is passed as revision
        self.processor = CLIPProcessor.from_pretrained(model_id, revision=model_revision)  # nosec B615
        self.model = CLIPModel.from_pretrained(model_id, revision=model_revision)  # nosec B615
        self.model.to(device)
        self.device = device

        # Trained classifier
        self.predictor = TorchPredictor.from_artifacts_dir(artifacts_dir=artifacts_dir)
        self.preprocessor = self.predictor.preprocessor

    @torch.inference_mode  # type: ignore
    def get_probabilities(self, url: str) -> dict[str, dict[str, float]]:
        image = load_image_from_url(url)
        inputs = self.processor(images=[image], return_tensors="pt", padding=True).to(self.device)
        embedding = self.model.get_image_features(**inputs).cpu().numpy()
        outputs = self.predictor.predict_probabilities(collate_fn({"embedding": embedding}))
        probabilities: dict[str, float] = outputs["probabilities"][0]
        return {"probabilities": probabilities}


@serve.deployment
@serve.ingress(api)
class IngressDeployment:
    def __init__(self, classifier: ClassifierModel):
        self.classifier = classifier

    @api.post("/predict")
    async def predict(self, request: Request) -> dict[str, dict[str, float]]:
        data = await request.json()
        probabilities = await self.classifier.get_probabilities.remote(url=data["image_url"])  # type: ignore[attr-defined]
        return probabilities


classifier_deployment = ClassifierModel.bind(  # type: ignore[attr-defined]
    model_id="openai/clip-vit-base-patch32",
    artifacts_dir=get_artifacts_dir_best_run(),
    device="cpu",
)

serve_app = IngressDeployment.bind(classifier=classifier_deployment)  # type: ignore[attr-defined]

if __name__ == "__main__":
    # Run service locally
    try:
        logger.info("Starting Dog Classifier service. Press Ctrl+C to exit.")
        serve.run(serve_app, route_prefix="/")
        time.sleep(600)
        logger.info("Serve run complete")
    except KeyboardInterrupt:
        logger.info("Service stopped by user.")
    finally:
        logger.info("Calling serve.shutdown")
        serve.shutdown()
