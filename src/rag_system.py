import time
from typing import Dict, Any, List
from src.config_loader import settings
from src.logger_setup import setup_logging
from src.rag_backends import VectorStoreBackend, GeneratorBackend
from src.retrieval_strategies import RetrievalStrategy
from src.utils import metrics_collector

logger = setup_logging(__name__)

class HSNRAGSystem:
    def __init__(self, vector_store: VectorStoreBackend, generator: GeneratorBackend, 
                 retrieval_strategy: RetrievalStrategy):
        self.vector_store = vector_store
        self.generator = generator
        self.retrieval_strategy = retrieval_strategy
        self.query_cache = {}

    async def initialize_vector_store(self, documents: List[Dict[str, Any]]):
        await self.vector_store.initialize(documents)

    def _build_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        context = "\n---\n".join([f"HSN Code: {doc['metadata'].get('hsn_code', 'N/A')}\nDescription: {doc['text']}\nGraph Context: {doc.get('graph_context', 'N/A')}" for doc in context_docs])
        prompt = f"""
        User query: "{query}"

        Based on the following retrieved HSN code information, provide a structured answer.
        - Classify the user's query with the most likely HSN code.
        - Provide a confidence score (High, Medium, Low).
        - Explain your reasoning based on the provided context.
        - List the top 3 potential matches with their descriptions.

        Context:
        {context}
        """
        return prompt

    async def retrieve_documents(self, query: str) -> List[Dict[str, Any]]:
        """Performs the retrieval step only and returns the documents."""
        retrieval_start = time.perf_counter()
        retrieved_docs = await self.retrieval_strategy.retrieve(query, self.vector_store)
        retrieval_time = (time.perf_counter() - retrieval_start) * 1000
        metrics_collector.record("retrieval_time_ms", retrieval_time)
        return retrieved_docs

    async def generate_from_docs(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Takes retrieved documents and performs the generation step."""
        prompt = self._build_prompt(query, retrieved_docs)
        generation_start = time.perf_counter()
        generated_text = await self.generator.generate_response(prompt)
        generation_time = (time.perf_counter() - generation_start) * 1000
        metrics_collector.record("generation_time_ms", generation_time)
        
        response = self.generate_structured_response(generated_text, retrieved_docs)
        return response

    def generate_structured_response(self, generated_text: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "summary": generated_text,
            "top_matches": [
                {
                    "hsn_code": doc['metadata'].get('hsn_code'),
                    "description": doc['metadata'].get('item_description'),
                    "full_context": doc['text'],
                    "retrieval_score": doc.get('score'),
                    "graph_context": doc.get('graph_context'),
                    "metadata": doc.get('metadata', {})  # Include metadata for test compatibility
                } for doc in context_docs
            ],
            "confidence": "High" if context_docs and context_docs[0].get('score', 0) > 0.85 else "Medium",
            "trade_policy": "Free"
        }