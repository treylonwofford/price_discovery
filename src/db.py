"""
Module instantiating the db
"""

# Import the necessary libraries
from typing import Union, Optional, Any
from qdrant_client import QdrantClient
from src.utils import logger
from langchain_community.vectorstores import Chroma

# Instantiate DB class
class VectorStore:
    """
    A class for managing vector store operations.

    Args:
        persist_directory (str): The directory where the vector store data will be persisted.
        embedding_function (Callable): The function used for embedding text or images.
        collection_name (str): The name of the collection in the vector store.
        remote (bool): Whether the vector store is remote or local.
        url (str, optional): The URL of the remote vector store (if remote is True).
        api_key (str, optional): The API key for accessing the remote vector store (if remote is True).

    Attributes:
        persist_directory (str): The directory where the vector store data will be persisted.
        embedding_function (Callable): The function used for embedding text or images.
        collection_name (str): The name of the collection in the vector store.
        remote (bool): Whether the vector store is remote or local.
        url (str): The URL of the remote vector store (if remote is True).
        api_key (str): The API key for accessing the remote vector store (if remote is True).
    """

    def __init__(
        self,
        persist_directory: Optional[str],
        embedding_function: callable,
        collection_name: str,
        remote: bool,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self.remote = remote
        self.url = url
        self.api_key = api_key

    def instantiate(self) -> Any:
        """
        Instantiate a Chroma vector store.

        Args:
            persist_directory (str): The directory where the vector store data will be persisted.
            embedding_function (Callable): The function used for embedding text or images.
            collection_name (str): The name of the collection in the vector store.
            remote (bool): Whether the vector store is remote or local.

        Returns:
            Chroma: The instantiated Chroma vector store.
        """
        logger.info(f"Instantiating Chroma vector store in {self.persist_directory}")
        if not self.remote:
            db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function,
                collection_name=self.collection_name,
            )
            return db
        else:
            client = QdrantClient(self.url, api_key=self.api_key)

            return client

    def add_data(self, db: Chroma, data: Union[list, str], data_type: str) -> None:
        """
        Add data (text or images) to the Chroma vector store.

        Args:
            db (Chroma): The instantiated Chroma vector store.
            data (Union[list, str]): The data to be added to the vector store (list of texts or images, or a single text).
            data_type (str): The type of data ('text' or 'images').
        """
        logger.info(f"Adding {data_type} data to the vector store")
        if data_type == "text":
            db.add_documents(data)
        else:
            db.add_images(data)

    def create_retriever(self, db: Chroma) -> Chroma.as_retriever:
        """
        Create a retriever from the Chroma vector store.

        Args:
            db (Chroma): The instantiated Chroma vector store.

        Returns:
            Chroma.as_retriever: The retriever for the Chroma vector store.
        """
        logger.info("Creating retriever from the vector store")
        return db.as_retriever()
