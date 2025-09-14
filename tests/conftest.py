import pytest
import pytest_asyncio
import yaml
from pathlib import Path
import json
import asyncio
from src.graph_backends import NetworkXBackend
from src.graph_builder import KnowledgeGraphBuilder
from src.rag_backends import ChromaBackend, MockGeneratorBackend
from src.retrieval_strategies import GraphContextualStrategy
from src.rag_system import HSNRAGSystem
from src.query_processor import HSNQueryProcessor

@pytest.fixture(scope="session")
def test_scenarios():
    """Loads test scenarios from the YAML file."""
    scenarios_path = Path(__file__).parent / "test_data/test_scenarios.yml"
    with open(scenarios_path, 'r') as f:
        return yaml.safe_load(f)["test_scenarios"]

@pytest_asyncio.fixture(scope="session")
async def initialized_query_processor():
    """A session-scoped fixture that initializes the entire system once for async tests."""
    docs_path = Path("data/processed/structured_hsn_documents.json")
    with open(docs_path, 'r') as f:
        documents = json.load(f)

    kg_backend = NetworkXBackend()
    kg_builder = KnowledgeGraphBuilder(kg_backend)
    kg_builder.load_documents(docs_path)
    kg_builder.build_hsn_knowledge_graph()

    vector_store = ChromaBackend()
    generator = MockGeneratorBackend()
    retrieval_strategy = GraphContextualStrategy(kg_builder)
    rag_system = HSNRAGSystem(vector_store, generator, retrieval_strategy)
    
    from src.config_loader import settings
    settings.rag_system.vector_store.path = "data/test_vector_store"
    await rag_system.initialize_vector_store(documents)

    query_processor = HSNQueryProcessor(rag_system)
    
    yield query_processor

    import shutil
    shutil.rmtree(settings.rag_system.vector_store.path, ignore_errors=True)

@pytest.fixture(scope="session")
def sync_initialized_query_processor():
    """
    A synchronous, self-contained fixture for benchmark tests.
    """
    docs_path = Path("data/processed/structured_hsn_documents.json")
    with open(docs_path, 'r') as f:
        documents = json.load(f)

    kg_backend = NetworkXBackend()
    kg_builder = KnowledgeGraphBuilder(kg_backend)
    kg_builder.load_documents(docs_path)
    kg_builder.build_hsn_knowledge_graph()

    vector_store = ChromaBackend()
    generator = MockGeneratorBackend()
    retrieval_strategy = GraphContextualStrategy(kg_builder)
    rag_system = HSNRAGSystem(vector_store, generator, retrieval_strategy)
    
    from src.config_loader import settings
    benchmark_db_path = "data/benchmark_vector_store"
    settings.rag_system.vector_store.path = benchmark_db_path
    
    asyncio.run(rag_system.initialize_vector_store(documents))

    query_processor = HSNQueryProcessor(rag_system)
    
    yield query_processor

    import shutil
    shutil.rmtree(benchmark_db_path, ignore_errors=True)