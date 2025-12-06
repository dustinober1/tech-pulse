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
    """

    def __init__(self,
                 collection_name: str = "tech_stories",
                 model_name: str = "all-MiniLM-L6-v2",
                 persist_directory: str = "./chroma_db",
                 device: str = "cpu"):
        """
        Initialize the VectorSearchManager.
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

    @property
    def embedding_model(self) -> SentenceTransformer:
        """
        Lazily load and return the sentence transformer model.
        """
        if self._embedding_model is None:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            try:
                # Show progress bar during model loading
                with tqdm(total=100, desc="Loading embedding model", unit="%") as pbar:
                    pbar.set_description(f"Downloading {self.model_name}")
                    self._embedding_model = SentenceTransformer(self.model_name, device=self.device)
                    pbar.update(50)
                    pbar.set_description("Initializing model")
                    # This triggers the full model initialization
                    self._embedding_model.encode(["test"], show_progress_bar=False)
                    pbar.update(50)
            except Exception as e:
                self.logger.error(f"Failed to load embedding model: {str(e)}")
                raise RuntimeError(f"Could not load embedding model {self.model_name}: {str(e)}")

        return self._embedding_model

    @property
    def chroma_client(self) -> chromadb.Client:
        """
        Lazily load and return the ChromaDB client.
        """
        if self._chroma_client is None:
            try:
                self.logger.info("Initializing ChromaDB client")
                self._chroma_client = chromadb.PersistentClient(path=self.persist_directory)
            except Exception as e:
                self.logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
                raise RuntimeError(f"Could not initialize ChromaDB client: {str(e)}")

        return self._chroma_client

    @property
    def collection(self) -> chromadb.Collection:
        """
        Get or create the ChromaDB collection.
        """
        if self._collection is None:
            try:
                self._collection = self.chroma_client.get_collection(name=self.collection_name)
            except Exception:
                self._collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Tech stories vector embeddings"}
                )

        return self._collection

    @collection.setter
    def collection(self, value):
        """Allow manual setting of collection (e.g. from cache)"""
        self._collection = value

    def initialize(self):
        """Explicit initialization method."""
        _ = self.chroma_client
        _ = self.collection
        _ = self.embedding_model

    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        """
        if not texts:
            return []

        try:
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
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
            return True
        except Exception as e:
            self.logger.error(f"Failed to add documents: {str(e)}")
            return False

    def search(self,
               query: str,
               n_results: int = 10,
               similarity_threshold: Optional[float] = None,
               where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search for similar documents.
        """
        try:
            query_embedding = self.embedding_model.encode([query])[0].tolist()

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where
            )

            formatted_results = []
            if results['ids']:
                for i, doc_id in enumerate(results['ids'][0]):
                    # Check threshold if provided
                    # Note: Chroma returns distances. Lower is better.
                    # Assuming default L2 distance.
                    # This logic was handled in data_loader, but good to have here too if needed.

                    formatted_results.append({
                        'id': doc_id,
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if 'distances' in results else None,
                        'similarity': 1.0 - (results['distances'][0][i] if 'distances' in results else 1.0) # Approximation
                    })

            return {
                'query': query,
                'results': formatted_results,
                'count': len(formatted_results)
            }
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            return {'query': query, 'results': [], 'count': 0, 'error': str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        try:
            count = self.collection.count()
            return {
                'collection_name': self.collection_name,
                'document_count': count,
                'model_name': self.model_name
            }
        except Exception as e:
            return {'error': str(e)}

    def delete_collection(self) -> bool:
        """Delete the entire collection."""
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            self._collection = None
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete collection: {str(e)}")
            return False


# Global instance
_vector_engine: Optional[VectorSearchManager] = None


def get_vector_engine(model_name: str = "all-MiniLM-L6-v2", collection_name: str = "hn_stories") -> VectorSearchManager:
    """Get or create a global vector search engine instance."""
    global _vector_engine
    if _vector_engine is None:
        _vector_engine = VectorSearchManager(
            collection_name=collection_name,
            model_name=model_name
        )
    return _vector_engine
