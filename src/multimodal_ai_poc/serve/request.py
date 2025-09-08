import asyncio

import aiohttp
from loguru import logger


class RequestException(Exception): ...


async def predict_image(
    image_url: str, service_url: str = "http://localhost:8000/predict"
) -> dict[str, dict[str, float]]:
    """
    Make an async POST request to the dog classifier service.

    Args:
        image_url: URL of the image to classify
        service_url: URL of the classifier service endpoint

    Returns:
        Dictionary containing the prediction results
    """
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url=service_url,
            json={"image_url": image_url},
            headers={"Content-Type": "application/json"},
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RequestException(f"Request failed with {response.status=}: {error_text}")
            return await response.json()


async def main():
    # Example usage
    image_url = "https://doggos-dataset.s3.us-west-2.amazonaws.com/samara.png"
    result = await predict_image(image_url)
    # Sort results
    probabilities = sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True)
    logger.info(f"Prediction probabilities: {probabilities}")


if __name__ == "__main__":
    asyncio.run(main())
