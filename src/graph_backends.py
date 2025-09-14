from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import networkx as nx
from neo4j import GraphDatabase, Driver
from src.logger_setup import setup_logging

logger = setup_logging(__name__)

class GraphBackend(ABC):
    """Abstract Base Class defining the interface for a graph backend."""

    @abstractmethod
    def add_node(self, node_id: str, properties: Dict[str, Any]):
        pass

    @abstractmethod
    def add_edge(self, source_id: str, target_id: str, rel_type: str, properties: Dict[str, Any] = None):
        pass

    @abstractmethod
    def create_indexes(self):
        pass

    @abstractmethod
    def get_neighbors(self, node_id: str, direction: str = 'out') -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_subgraph(self, node_id: str, depth: int = 1) -> Any:
        pass

    @abstractmethod
    def export_to_graphml(self, file_path: str):
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, int]:
        pass

    @abstractmethod
    def close(self):
        pass


class NetworkXBackend(GraphBackend):
    """NetworkX implementation of the GraphBackend interface."""

    def __init__(self):
        logger.info("Initializing NetworkX backend.")
        self.graph = nx.DiGraph()

    def add_node(self, node_id: str, properties: Dict[str, Any]):
        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, **properties)

    def add_edge(self, source_id: str, target_id: str, rel_type: str, properties: Dict[str, Any] = None):
        if not self.graph.has_edge(source_id, target_id):
            self.graph.add_edge(source_id, target_id, type=rel_type, **(properties or {}))

    def create_indexes(self):
        logger.info("NetworkX backend does not require explicit index creation. Operations are in-memory.")
        pass

    def get_neighbors(self, node_id: str, direction: str = 'out') -> List[Dict[str, Any]]:
        if direction == 'out':
            neighbors = self.graph.successors(node_id)
        else:
            neighbors = self.graph.predecessors(node_id)
        return [self.graph.nodes[n] for n in neighbors]

    def get_subgraph(self, node_id: str, depth: int = 1) -> nx.DiGraph:
        nodes = {node_id}
        for _ in range(depth):
            new_nodes = set()
            for node in nodes:
                new_nodes.update(self.graph.successors(node))
                new_nodes.update(self.graph.predecessors(node))
            nodes.update(new_nodes)
        return self.graph.subgraph(nodes)

    def export_to_graphml(self, file_path: str):
        logger.info(f"Exporting graph to GraphML at {file_path}")
        nx.write_graphml(self.graph, file_path)

    def get_statistics(self) -> Dict[str, int]:
        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges()
        }

    def close(self):
        logger.info("Closing NetworkX backend (in-memory, no action needed).")
        pass


class Neo4jBackend(GraphBackend):
    """Neo4j implementation of the GraphBackend interface."""

    def __init__(self, uri: str, user: str, password: str):
        logger.info(f"Initializing Neo4j backend for URI: {uri}")
        self.driver: Driver = GraphDatabase.driver(uri, auth=(user, password))

    def add_node(self, node_id: str, properties: Dict[str, Any]):
        label = properties.get("label", "Node")
        with self.driver.session() as session:
            query = f"""
            MERGE (n:{label} {{id: $node_id}})
            SET n += $properties
            """
            session.run(query, node_id=node_id, properties=properties)

    def add_edge(self, source_id: str, target_id: str, rel_type: str, properties: Dict[str, Any] = None):
        with self.driver.session() as session:
            query = f"""
            MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
            MERGE (a)-[r:{rel_type}]->(b)
            SET r += $properties
            """
            session.run(query, source_id=source_id, target_id=target_id, properties=(properties or {}))

    def create_indexes(self):
        logger.info("Creating indexes in Neo4j for performance optimization...")
        with self.driver.session() as session:
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:Chapter) ON (n.id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:Heading) ON (n.id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:Subheading) ON (n.id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:HSNCode) ON (n.id)")
        logger.info("Neo4j indexes created successfully.")

    def get_neighbors(self, node_id: str, direction: str = 'out') -> List[Dict[str, Any]]:
        arrow = "->" if direction == 'out' else "<-"
        with self.driver.session() as session:
            query = f"MATCH (a {{id: $node_id}}){arrow}(b) RETURN b"
            results = session.run(query, node_id=node_id)
            return [record["b"] for record in results]

    def get_subgraph(self, node_id: str, depth: int = 1) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            query = f"""
            MATCH path = (n {{id: $node_id}})-[*1..{depth}]-(m)
            RETURN path
            """
            results = session.run(query, node_id=node_id)
            return [record["path"] for record in results]

    def export_to_graphml(self, file_path: str):
        logger.warning("Direct GraphML export from Neo4j driver is complex. Use APOC library in Neo4j instead.")
        logger.info("To enable, run: CALL apoc.export.graphml.all({file}, {{}})")
        pass

    def get_statistics(self) -> Dict[str, int]:
        with self.driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
            edge_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
            return {"node_count": node_count, "edge_count": edge_count}

    def close(self):
        logger.info("Closing Neo4j driver connection.")
        self.driver.close()