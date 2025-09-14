from abc import ABC, abstractmethod
from typing import List, Dict, Any
from async_lru import alru_cache
import networkx as nx
from src.rag_backends import VectorStoreBackend
from src.graph_builder import KnowledgeGraphBuilder
from src.config_loader import settings
from src.graph_backends import NetworkXBackend
from sentence_transformers import CrossEncoder

class RetrievalStrategy(ABC):
    @abstractmethod
    async def retrieve(self, query: str, vector_store: VectorStoreBackend, **kwargs) -> List[Dict[str, Any]]:
        pass

class VectorOnlyStrategy(RetrievalStrategy):
    async def retrieve(self, query: str, vector_store: VectorStoreBackend, **kwargs) -> List[Dict[str, Any]]:
        return await vector_store.query(query, top_k=settings.rag_system.retrieval.top_k)

class ReRankStrategy(RetrievalStrategy):
    def __init__(self):
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.candidate_multiplier = 4

    async def retrieve(self, query: str, vector_store: VectorStoreBackend, **kwargs) -> List[Dict[str, Any]]:
        top_k = settings.rag_system.retrieval.top_k
        num_candidates = top_k * self.candidate_multiplier

        candidate_docs = await vector_store.query(query, top_k=num_candidates)
        if not candidate_docs:
            return []

        sentence_pairs = [[query, doc['text']] for doc in candidate_docs]
        scores = self.cross_encoder.predict(sentence_pairs)

        for doc, score in zip(candidate_docs, scores):
            doc['rerank_score'] = float(score)
            doc['score'] = float(score)

        reranked_docs = sorted(candidate_docs, key=lambda x: x['rerank_score'], reverse=True)
        return reranked_docs[:top_k]

class GraphContextualStrategy(RetrievalStrategy):
    def __init__(self, kg_builder: KnowledgeGraphBuilder):
        self.kg_builder = kg_builder
        self.base_retriever = ReRankStrategy()

    @alru_cache(maxsize=256)
    async def _get_graph_context(self, hsn_code: str) -> str:
        node_id = f"code_{hsn_code}"
        
        if not isinstance(self.kg_builder.backend, NetworkXBackend):
            return "Graph context not available for this backend."

        graph = self.kg_builder.backend.graph
        if not graph.has_node(node_id):
            return "HSN code not found in graph."

        path_nodes = []
        current_node = node_id
        while current_node:
            path_nodes.append(current_node)
            predecessors = list(graph.predecessors(current_node))
            if not predecessors:
                break
            current_node = predecessors[0]

        context_parts = []
        for node in reversed(path_nodes):
            node_attrs = graph.nodes[node]
            label = node_attrs.get('label', 'Unknown')
            desc = node_attrs.get('description', 'N/A')
            
            if label != 'HSNCode':
                context_parts.append(f"{label}: {desc}")

        return ". ".join(context_parts)

    async def retrieve(self, query: str, vector_store: VectorStoreBackend, **kwargs) -> List[Dict[str, Any]]:
        initial_results = await self.base_retriever.retrieve(query, vector_store)
        
        enhanced_results = []
        for res in initial_results:
            hsn_code = res['metadata'].get('hsn_code')
            if hsn_code:
                graph_context = await self._get_graph_context(hsn_code)
                res['graph_context'] = graph_context
            enhanced_results.append(res)
            
        return enhanced_results