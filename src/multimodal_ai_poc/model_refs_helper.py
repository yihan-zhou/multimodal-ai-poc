from huggingface_hub import HfApi
from loguru import logger


def main():
    api = HfApi()
    refs = api.list_repo_refs("openai/clip-vit-base-patch32")
    logger.info(refs)


if __name__ == "__main__":
    main()
