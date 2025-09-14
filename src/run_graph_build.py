from pathlib import Path
import sys
from src.config_loader import settings
from src.logger_setup import setup_logging
from src.graph_backends import NetworkXBackend, Neo4jBackend
from src.graph_builder import KnowledgeGraphBuilder

logger = setup_logging(__name__)

def get_backend():
    """Factory function to instantiate the correct backend based on config."""
    backend_choice = settings.knowledge_graph.backend.lower()
    if backend_choice == "networkx":
        return NetworkXBackend()
    elif backend_choice == "neo4j":
        try:
            return Neo4jBackend(
                uri=settings.knowledge_graph.neo4j.uri,
                user=settings.knowledge_graph.neo4j.user,
                password=settings.knowledge_graph.neo4j.password
            )
        except Exception as e:
            logger.critical(f"Failed to connect to Neo4j. Check credentials and connection. Error: {e}")
            sys.exit(1)
    else:
        raise ValueError(f"Unsupported backend: {backend_choice}")

def run_graph_pipeline():
    """Executes the full Knowledge Graph construction and validation pipeline."""
    logger.info("Starting HSN Knowledge Graph Pipeline...")
    logger.info(f"Using backend: {settings.knowledge_graph.backend}")

    backend = get_backend()
    builder = KnowledgeGraphBuilder(backend)

    try:
        docs_path = Path(settings.data_paths.processed_docs)
        builder.load_documents(docs_path)
        builder.build_hsn_knowledge_graph()
        builder.enrich_with_semantic_relationships(settings.knowledge_graph.llm_enrichment.dict())
        builder.optimize_graph_structure()
        builder.validate_graph_integrity()
        stats = builder.generate_graph_statistics()
        logger.info(f"Graph Statistics: {stats}")
        export_path = Path(settings.knowledge_graph.export_path)
        builder.export_graph_data(export_path)
        viz_path = Path(settings.knowledge_graph.visualization_path)
        builder.visualize_graph_structure(viz_path)
        logger.info("HSN Knowledge Graph Pipeline completed successfully.")

    except Exception as e:
        logger.critical(f"An unhandled exception occurred in the graph pipeline: {e}", exc_info=True)
        sys.exit(1)
    finally:
        backend.close()

if __name__ == "__main__":
    run_graph_pipeline()