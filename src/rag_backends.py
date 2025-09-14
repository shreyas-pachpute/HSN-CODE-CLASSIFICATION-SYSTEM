import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from pybreaker import CircuitBreakerError

from src.config_loader import settings
from src.logger_setup import setup_logging
from src.utils import CircuitBreakerManager

logger = setup_logging(__name__)

class VectorStoreBackend(ABC):
    @abstractmethod
    async def initialize(self, documents: List[Dict[str, Any]]):
        pass
    
    @abstractmethod
    async def query(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        pass

class ChromaBackend(VectorStoreBackend):
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.rag_system.vector_store.path)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.rag_system.vector_store.embedding_model
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.rag_system.vector_store.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"} 
        )

    async def initialize(self, documents: List[Dict[str, Any]]):
        logger.info(f"Initializing ChromaDB with {len(documents)} documents...")
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            self.collection.add(
                ids=[doc['document_id'] for doc in batch],
                documents=[doc['text'] for doc in batch],
                metadatas=[doc['metadata'] for doc in batch]
            )
        logger.info("ChromaDB initialization complete.")

    async def query(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=["metadatas", "documents", "distances"]
        )
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': 1 - results['distances'][0][i] 
            })
        return formatted_results

class GeneratorBackend(ABC):
    @abstractmethod
    async def generate_response(self, prompt: str) -> str:
        pass

class MockGeneratorBackend(GeneratorBackend):
    """A mock generator for development and testing without API calls."""
    async def generate_response(self, prompt: str) -> str:
        logger.info("Using MockGeneratorBackend.")
        await asyncio.sleep(0.1)
        return f"Mock response based on the following context:\n---\n{prompt[:500]}...\n---"

class GeminiGeneratorBackend(GeneratorBackend):
    """A generator backend using Google's Gemini models."""
    def __init__(self):
        gemini_api_key = settings.rag_system.generator.gemini_api_key
        if not gemini_api_key:
            raise ValueError("Gemini API key is not configured in settings.")
        
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(settings.rag_system.generator.model)
        logger.info(f"Initialized GeminiGeneratorBackend with model: {settings.rag_system.generator.model}")

    @CircuitBreakerManager.llm_breaker
    async def generate_response(self, prompt: str) -> str:
        logger.info(f"Calling Gemini API with model {self.model.model_name}...")
        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=settings.rag_system.generator.temperature
                )
            )
            return response.text
        except CircuitBreakerError as e:
            logger.error(f"Circuit breaker is open for LLM calls: {e}")
            return "The system is currently experiencing high load. Please try again later."
        except Exception as e:
            logger.error(f"Gemini API error: {e}", exc_info=True)
            raise