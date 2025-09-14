import asyncio
import json
from pathlib import Path
from src.config_loader import settings
from src.logger_setup import setup_logging
from src.graph_backends import NetworkXBackend
from src.graph_builder import KnowledgeGraphBuilder
from src.rag_backends import ChromaBackend
from src.rag_system import HSNRAGSystem
from src.conversation_manager import ConversationState
from src.query_processor import HSNQueryProcessor

logger = setup_logging(__name__)

def setup_generator():
    backend = settings.rag_system.generator.backend
    if backend == "gemini":
        from src.rag_backends import GeminiGeneratorBackend
        return GeminiGeneratorBackend()
    from src.rag_backends import MockGeneratorBackend
    return MockGeneratorBackend()

def setup_retrieval_strategy(kg_builder: KnowledgeGraphBuilder):
    strategy = settings.rag_system.retrieval.strategy
    if strategy == "vector":
        from src.retrieval_strategies import VectorOnlyStrategy
        return VectorOnlyStrategy()
    if strategy == "graph_contextual":
        from src.retrieval_strategies import GraphContextualStrategy
        return GraphContextualStrategy(kg_builder)
    raise ValueError(f"Unknown retrieval strategy: {strategy}")

async def main():
    logger.info("--- Initializing Full HSN Classification System ---")
    docs_path = Path(settings.data_paths.processed_docs)
    with open(docs_path, 'r') as f:
        documents = json.load(f)

    kg_backend = NetworkXBackend()
    kg_builder = KnowledgeGraphBuilder(kg_backend)
    kg_builder.load_documents(docs_path)
    kg_builder.build_hsn_knowledge_graph()
    logger.info("Knowledge Graph loaded.")
    vector_store = ChromaBackend()
    generator = setup_generator() 
    retrieval_strategy = setup_retrieval_strategy(kg_builder) 
    rag_system = HSNRAGSystem(vector_store, generator, retrieval_strategy)
    await rag_system.initialize_vector_store(documents)
    logger.info("RAG System initialized and Vector Store populated.")
    query_processor = HSNQueryProcessor(rag_system)
    conversation = ConversationState()
    logger.info("--- System Ready for Interactive Session ---")
    print("\nWelcome to the HSN Code Classification Assistant. Type 'exit' to end.")

    while True:
        user_input = input("\nYour query: ")
        if user_input.lower() == 'exit':
            print("Session ended. Goodbye!")
            break
        
        response = await query_processor.process_query(user_input, conversation)
        print(f"\nSystem: {response['summary']}")

if __name__ == "__main__":
    asyncio.run(main())