import logging
from typing import Union, Optional, cast

import httpx

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.client_utils import get_client_options

logger = logging.getLogger(__name__)


class OllamaEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    This class is used to generate embeddings for a list of texts using the Ollama Embedding API (https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings).
    """

    def __init__(
        self, url: str, model_name: str, timeout: Optional[float] = None
    ) -> None:
        """
        Initialize the Ollama Embedding Function.

        Args:
            url (str): The URL of the Ollama Server.
            model_name (str): The name of the model to use for text embeddings. E.g. "nomic-embed-text" (see https://ollama.com/library for available models).
            timeout (Optional[float]): The timeout for the Ollama http client. If fot specified the DEFAULT_TIMEOUT_CONFIG is set as the default timeout.
        """
        self._api_url = f"{url}"
        self._model_name = model_name
        client_options = get_client_options(timeout=timeout)
        self._session = httpx.Client(**client_options)

    def __call__(self, input: Union[Documents, str]) -> Embeddings:
        """
        Get the embeddings for a list of texts.

        Args:
            input (Documents): A list of texts to get embeddings for.

        Returns:
            Embeddings: The embeddings for the texts.

        Example:
            >>> ollama_ef = OllamaEmbeddingFunction(url="http://localhost:11434/api/embeddings", model_name="nomic-embed-text")
            >>> texts = ["Hello, world!", "How are you?"]
            >>> embeddings = ollama_ef(texts)
        """
        # Call Ollama Server API for each document
        texts = input if isinstance(input, list) else [input]
        embeddings = [
            self._session.post(
                self._api_url, json={"model": self._model_name, "prompt": text}
            ).json()
            for text in texts
        ]
        return cast(
            Embeddings,
            [
                embedding["embedding"]
                for embedding in embeddings
                if "embedding" in embedding
            ],
        )
