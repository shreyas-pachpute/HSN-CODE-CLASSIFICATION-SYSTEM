import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from functools import lru_cache
import networkx as nx
from src.graph_backends import GraphBackend
from src.logger_setup import setup_logging
from src.performance_monitor import performance_metric

logger = setup_logging(__name__)

class KnowledgeGraphBuilder:
    """
    Builds, enriches, and operates on an HSN Code Knowledge Graph.
    This class is backend-agnostic, operating on an injected GraphBackend instance.
    """

    def __init__(self, backend: GraphBackend):
        """
        Initializes the builder with a specific graph backend.

        Args:
            backend (GraphBackend): An instance of a class that implements the GraphBackend interface.
        """
        self.backend = backend
        self.documents: List[Dict[str, Any]] = []

    def load_documents(self, file_path: Path):
        """Loads the structured documents from the data processing pipeline."""
        logger.info(f"Loading structured documents from {file_path}...")
        with open(file_path, 'r') as f:
            self.documents = json.load(f)
        logger.info(f"Loaded {len(self.documents)} documents.")

    @performance_metric
    def build_hsn_knowledge_graph(self):
        """
        Constructs the foundational hierarchical graph from loaded documents.
        Creates nodes for Chapters, Headings, Subheadings, and HSN Codes.
        """
        if not self.documents:
            raise ValueError("Documents not loaded. Call load_documents() first.")
        
        logger.info("Building the hierarchical HSN knowledge graph...")
        for doc in self.documents:
            meta = doc['metadata']
            self.add_entity_relationships(meta)
        
        logger.info("Hierarchical graph construction complete.")
        return self 

    def add_entity_relationships(self, metadata: Dict[str, Any]):
        """
        Adds nodes and edges for a single HSN code's hierarchy.
        This method is idempotent; it won't create duplicate nodes or edges.
        """
        chapter_id = f"chap_{metadata['chapter']}"
        heading_id = f"head_{metadata['heading']}"
        subheading_id = f"sub_{metadata['subheading']}"
        code_id = f"code_{metadata['hsn_code']}"
        chapter_desc = metadata.get('chapter_description', 'N/A')
        heading_desc = metadata.get('heading_description', 'N/A')
        subheading_desc = metadata.get('subheading_description', 'N/A')
        item_desc = metadata.get('item_description', 'N/A')
        self.backend.add_node(chapter_id, {"id": chapter_id, "description": chapter_desc, "label": "Chapter"})
        self.backend.add_node(heading_id, {"id": heading_id, "description": heading_desc, "label": "Heading"})
        self.backend.add_node(subheading_id, {"id": subheading_id, "description": subheading_desc, "label": "Subheading"})
        self.backend.add_node(code_id, {"id": code_id, "description": item_desc, "label": "HSNCode"})
        self.backend.add_edge(chapter_id, heading_id, "HAS_HEADING")
        self.backend.add_edge(heading_id, subheading_id, "HAS_SUBHEADING")
        self.backend.add_edge(subheading_id, code_id, "HAS_CODE")

    @performance_metric
    def enrich_with_semantic_relationships(self, llm_config: Dict[str, Any]):
        """
        Enriches the graph with non-hierarchical relationships.
        - Rule-based: Adds 'SIBLING_OF' relationships between codes under the same subheading.
        - LLM-assisted: Adds 'SIMILAR_TO' relationships based on description similarity.
        """
        logger.info("Enriching graph with semantic relationships...")
        codes_by_subheading = {}
        for doc in self.documents:
            meta = doc['metadata']
            sub_id = f"sub_{meta['subheading']}"
            if sub_id not in codes_by_subheading:
                codes_by_subheading[sub_id] = []
            codes_by_subheading[sub_id].append(f"code_{meta['hsn_code']}")
        
        for sub_id, codes in codes_by_subheading.items():
            if len(codes) > 1:
                for i in range(len(codes)):
                    for j in range(i + 1, len(codes)):
                        self.backend.add_edge(codes[i], codes[j], "SIBLING_OF")
                        self.backend.add_edge(codes[j], codes[i], "SIBLING_OF")
        logger.info("Added rule-based 'SIBLING_OF' relationships.")
        if llm_config.get("enabled", False):
            self._add_llm_similarity_edges(llm_config)
        return self

    def _add_llm_similarity_edges(self, llm_config: Dict[str, Any]):
        """Helper for LLM-based enrichment."""
        logger.info("Starting LLM-based similarity enrichment...")
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            logger.error("Required libraries for LLM enrichment not found. Please run 'pip install sentence-transformers scikit-learn'.")
            return

        model = SentenceTransformer(llm_config['embedding_model'])
        item_docs = [doc for doc in self.documents if doc['metadata'].get('hsn_code')]
        
        if not item_docs:
            logger.warning("No item documents found for LLM enrichment.")
            return

        descriptions = [doc['text'] for doc in item_docs]
        ids = [f"code_{doc['metadata']['hsn_code']}" for doc in item_docs]
        
        logger.info(f"Generating embeddings for {len(descriptions)} descriptions...")
        embeddings = model.encode(descriptions, show_progress_bar=True)
        
        logger.info("Calculating similarity matrix...")
        sim_matrix = cosine_similarity(embeddings)

        threshold = llm_config['similarity_threshold']
        for i in range(len(sim_matrix)):
            for j in range(i + 1, len(sim_matrix)):
                if sim_matrix[i, j] > threshold:
                    logger.debug(f"Found similarity between {ids[i]} and {ids[j]}: {sim_matrix[i,j]:.4f}")
                    self.backend.add_edge(ids[i], ids[j], "SIMILAR_TO", {"score": float(sim_matrix[i, j])})

        logger.info("LLM-based enrichment complete.")

    def optimize_graph_structure(self):
        """Triggers backend-specific optimizations like index creation."""
        self.backend.create_indexes()
        return self

    @lru_cache(maxsize=128)
    def traverse_hierarchy(self, hsn_code: str, direction: str = 'up') -> List[Dict[str, Any]]:
        """
        Traverses the graph hierarchy from a given HSN code.
        Uses LRU caching for performance on repeated queries.

        Args:
            hsn_code (str): The 8-digit HSN code to start from.
            direction (str): 'up' for ancestors, 'down' for descendants.

        Returns:
            List[Dict[str, Any]]: A list of nodes in the traversal path.
        """
        node_id = f"code_{hsn_code}"
        traverse_dir = 'in' if direction == 'up' else 'out'
        return self.backend.get_neighbors(node_id, direction=traverse_dir)

    def get_context_subgraph(self, hsn_code: str, depth: int = 1) -> Any:
        """Extracts a subgraph around a specific HSN code."""
        node_id = f"code_{hsn_code}"
        return self.backend.get_subgraph(node_id, depth)

    def validate_graph_integrity(self) -> bool:
        """
        Performs health checks on the graph.
        - Checks for disconnected HSN code nodes.
        """
        logger.info("Validating graph integrity...")
        is_valid = True
        for doc in self.documents:
            code_id = f"code_{doc['metadata']['hsn_code']}"
            parents = self.backend.get_neighbors(code_id, direction='in')
            if not any(p.get('label') == 'Subheading' for p in parents):
                 logger.warning(f"Integrity check failed: Node {code_id} has no Subheading parent.")
                 is_valid = False
        
        if is_valid:
            logger.info("Graph integrity validation passed.")
        return is_valid

    def visualize_graph_structure(self, output_path: Path, notebook: bool = False):
        """Creates an interactive HTML visualization of the graph."""
        if not hasattr(self.backend, 'graph') or not isinstance(self.backend.graph, nx.Graph):
            logger.warning("Visualization is only supported for the NetworkX backend in this implementation.")
            return

        from pyvis.network import Network
        logger.info(f"Generating interactive visualization at {output_path}...")
        net = Network(notebook=notebook, directed=True, height="800px", width="100%")
        net.from_nx(self.backend.graph)
        net.save_graph(str(output_path))
        logger.info("Visualization saved.")

    def export_graph_data(self, file_path: Path):
        """Exports the graph data using the backend's method."""
        self.backend.export_to_graphml(file_path)

    def generate_graph_statistics(self) -> Dict[str, int]:
        """Returns basic statistics about the graph."""
        return self.backend.get_statistics()