import asyncio
import json
from pathlib import Path
from src.config_loader import settings
from src.logger_setup import setup_logging
from src.graph_backends import NetworkXBackend
from src.graph_builder import KnowledgeGraphBuilder
from src.rag_backends import ChromaBackend, MockGeneratorBackend, GeminiGeneratorBackend
from src.retrieval_strategies import VectorOnlyStrategy, HybridStrategy, GraphContextualStrategy
from src.rag_system import HSNRAGSystem

logger = setup_logging(__name__)

def get_generator():
    backend = settings.rag_system.generator.backend
    if backend == "gemini":
        return GeminiGeneratorBackend()
    if backend == "mock":
        return MockGeneratorBackend()
    raise ValueError(f"Unknown generator backend: {backend}")

def get_retrieval_strategy(kg_builder: KnowledgeGraphBuilder):
    strategy = settings.rag_system.retrieval.strategy
    if strategy == "vector":
        return VectorOnlyStrategy()
    if strategy == "hybrid":
        return HybridStrategy()
    if strategy == "graph_contextual":
        return GraphContextualStrategy(kg_builder)
    raise ValueError(f"Unknown retrieval strategy: {strategy}")

async def main():
    logger.info("--- Setting up HSN RAG System ---")
    docs_path = Path(settings.data_paths.processed_docs)
    with open(docs_path, 'r') as f:
        documents = json.load(f)
    kg_backend = NetworkXBackend()
    kg_builder = KnowledgeGraphBuilder(kg_backend)
    kg_builder.load_documents(docs_path)
    kg_builder.build_hsn_knowledge_graph()
    vector_store = ChromaBackend()
    generator = get_generator()
    retrieval_strategy = get_retrieval_strategy(kg_builder)
    
    rag_system = HSNRAGSystem(vector_store, generator, retrieval_strategy)
    await rag_system.initialize_vector_store(documents)
    
    logger.info("--- System Ready. Running Queries ---")
    
    queries = [
        "natural rubber latex",
        "computer monitors not incorporating a television reception apparatus",
        "prevulcanised rubber sheets"
    ]
    
    for query in queries:
        print(f"\n--- Querying for: '{query}' ---")
        response = await rag_system.query_hsn(query)
        print(json.dumps(response, indent=2))

if __name__ == "__main__":
    asyncio.run(main())