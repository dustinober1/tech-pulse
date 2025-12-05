"""
Vector Search Manager Module for Tech Pulse Dashboard

This module provides vector search capabilities using ChromaDB and sentence transformers
for semantic search on Hacker News stories and other tech content.
"""

import logging
import time
from typing import List, Dict, Optional, Tuple, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm


class VectorSearchManager:
    """
    Manages vector search operations using ChromaDB and sentence transformers.

    This class provides:
    - Lazy loading of embedding models
    - ChromaDB client management
    - Vector storage and retrieval
    - Semantic search capabilities
    - Progress indicators for model loading
    """

    def __init__(self,
                 collection_name: str = "tech_stories",
                 model_name: str = "all-MiniLM-L6-v2",
                 persist_directory: str = "./chroma_db",
                 device: str = "cpu"):
        """
        Initialize the VectorSearchManager.

        Args:
            collection_name: Name for the ChromaDB collection
            model_name: Name of the sentence transformer model
            persist_directory: Directory to persist the ChromaDB
            device: Device to run embeddings on ('cpu' or 'cuda')
        """
        self.logger = logging.getLogger(__name__)
        self.collection_name = collection_name
        self.model_name = model_name
        self.persist_directory = persist_directory
        self.device = device

        # Initialize as None for lazy loading
        self._embedding_model = None
        self._chroma_client = None
        self._collection = None

        self.logger.info(f"VectorSearchManager initialized with model: {model_name}")

    @property
    def embedding_model(self) -> SentenceTransformer:
        """
        Lazily load and return the sentence transformer model.

        Returns:
            SentenceTransformer: The loaded embedding model
        """
        if self._embedding_model is None:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            start_time = time.time()

            try:
                # Show progress bar during model loading
                with tqdm(total=100, desc="Loading embedding model", unit="%") as pbar:
                    pbar.set_description(f"Downloading {self.model_name}")
                    self._embedding_model = SentenceTransformer(self.model_name, device=self.device)
                    pbar.update(50)

                    pbar.set_description("Initializing model")
                    # This triggers the full model initialization
                    test_embedding = self._embedding_model.encode(["test"], show_progress_bar=False)
                    pbar.update(50)

                load_time = time.time() - start_time
                self.logger.info(f"Model loaded successfully in {load_time:.2f} seconds")

            except Exception as e:
                self.logger.error(f"Failed to load embedding model: {str(e)}")
                raise RuntimeError(f"Could not load embedding model {self.model_name}: {str(e)}")

        return self._embedding_model

    @property
    def chroma_client(self) -> chromadb.Client:
        """
        Lazily load and return the ChromaDB client.

        Returns:
            chromadb.Client: The initialized ChromaDB client
        """
        if self._chroma_client is None:
            try:
                self.logger.info("Initializing ChromaDB client")
                self._chroma_client = chromadb.PersistentClient(path=self.persist_directory)
                self.logger.info(f"ChromaDB client initialized with persist directory: {self.persist_directory}")
            except Exception as e:
                self.logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
                raise RuntimeError(f"Could not initialize ChromaDB client: {str(e)}")

        return self._chroma_client

    @property
    def collection(self) -> chromadb.Collection:
        """
        Get or create the ChromaDB collection.

        Returns:
            chromadb.Collection: The collection for vector storage
        """
        if self._collection is None:
            try:
                # Try to get existing collection
                self._collection = self.chroma_client.get_collection(name=self.collection_name)
                self.logger.info(f"Connected to existing collection: {self.collection_name}")
            except Exception:
                # Create new collection if it doesn't exist
                self._collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Tech stories vector embeddings"}
                )
                self.logger.info(f"Created new collection: {self.collection_name}")

        return self._collection

    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        try:
            self.logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            self.logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings.tolist()

        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {str(e)}")
            raise RuntimeError(f"Embedding generation failed: {str(e)}")

    def add_documents(self,
                      documents: List[str],
                      ids: List[str],
                      metadata: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        Add documents to the vector store.

        Args:
            documents: List of document texts
            ids: List of unique document IDs
            metadata: Optional list of metadata dictionaries

        Returns:
            bool: True if successful, False otherwise
        """
        if len(documents) != len(ids):
            raise ValueError("Number of documents and IDs must match")

        try:
            # Generate embeddings
            embeddings = self.generate_embeddings(documents)

            # Prepare metadata if not provided
            if metadata is None:
                metadata = [{"text": doc[:100]} for doc in documents]

            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                ids=ids,
                metadatas=metadata
            )

            self.logger.info(f"Successfully added {len(documents)} documents to vector store")
            return True

        except Exception as e:
            self.logger.error(f"Failed to add documents: {str(e)}")
            return False

    def search(self,
               query: str,
               n_results: int = 10,
               where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search for similar documents.

        Args:
            query: Search query string
            n_results: Number of results to return
            where: Optional metadata filter

        Returns:
            Dictionary containing search results
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0].tolist()

            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where
            )

            # Format results
            formatted_results = []
            for i, doc_id in enumerate(results['ids'][0]):
                formatted_results.append({
                    'id': doc_id,
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })

            return {
                'query': query,
                'results': formatted_results,
                'count': len(formatted_results)
            }

        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            return {
                'query': query,
                'results': [],
                'count': 0,
                'error': str(e)
            }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            return {
                'collection_name': self.collection_name,
                'document_count': count,
                'model_name': self.model_name,
                'persist_directory': self.persist_directory,
                'device': self.device
            }
        except Exception as e:
            self.logger.error(f"Failed to get stats: {str(e)}")
            return {'error': str(e)}

    def delete_collection(self) -> bool:
        """
        Delete the entire collection.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            self._collection = None
            self.logger.info(f"Successfully deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete collection: {str(e)}")
            return False


# Convenience function for quick initialization
def create_vector_search_manager(collection_name: str = "tech_stories") -> VectorSearchManager:
    """
    Create a VectorSearchManager with default settings.

    Args:
        collection_name: Name for the collection

    Returns:
        VectorSearchManager: Initialized manager
    """
    return VectorSearchManager(collection_name=collection_name)


# Backward compatibility - keep the old class name as an alias
VectorSearchEngine = VectorSearchManager


# Global instance for reuse across the application
_vector_engine = None


def get_vector_engine(model_name: str = "all-MiniLM-L6-v2", collection_name: str = "hn_stories") -> VectorSearchManager:
    """
    Get or create a global vector search engine instance.

    Args:
        model_name: Name of the sentence transformer model
        collection_name: Name of the ChromaDB collection

    Returns:
        VectorSearchManager instance
    """
    global _vector_engine

    if _vector_engine is None:
        _vector_engine = VectorSearchManager(
            collection_name=collection_name,
            model_name=model_name
        )

    return _vector_engine