import os
from pathlib import Path
from typing import Dict, Any
import yaml
from pydantic import BaseModel, Field, FilePath

class DataPaths(BaseModel):
    raw_hsn_data: FilePath
    processed_docs: Path

class LoggingConfig(BaseModel):
    level: str = Field(..., pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_file: Path
    rotation: str
    retention: str

class DataValidationConfig(BaseModel):
    min_chapter_number: int = Field(..., gt=0)
    max_chapter_number: int = Field(..., lt=100)
    hsn_code_length: int = Field(..., gt=0)

class Neo4jConfig(BaseModel):
    uri: str
    user: str
    password: str

class LLMEnrichmentConfig(BaseModel):
    enabled: bool
    embedding_model: str
    similarity_threshold: float = Field(..., gt=0, lt=1)

class KnowledgeGraphConfig(BaseModel):
    backend: str = Field(..., pattern=r"^(networkx|neo4j)$")
    export_path: Path
    visualization_path: Path
    neo4j: Neo4jConfig
    llm_enrichment: LLMEnrichmentConfig
    
class VectorStoreConfig(BaseModel):
    backend: str
    path: str
    collection_name: str
    embedding_model: str

class RetrievalConfig(BaseModel):
    strategy: str
    top_k: int

class GeneratorConfig(BaseModel):
    backend: str
    model: str
    openai_api_key: str | None = None 
    gemini_api_key: str | None = None
    temperature: float = Field(..., ge=0, le=2)
    timeout: int

class CachingConfig(BaseModel):
    query_cache_ttl: int

class CircuitBreakerConfig(BaseModel):
    fail_max: int
    reset_timeout: int

class RAGSystemConfig(BaseModel):
    vector_store: VectorStoreConfig
    retrieval: RetrievalConfig
    generator: GeneratorConfig
    caching: CachingConfig
    circuit_breaker: CircuitBreakerConfig

class Settings(BaseModel):
    """Main configuration model for the application."""
    data_paths: DataPaths
    logging: LoggingConfig
    data_validation: DataValidationConfig
    knowledge_graph: KnowledgeGraphConfig
    rag_system: RAGSystemConfig 

def _load_config_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_settings() -> Settings:
    env = os.getenv("APP_ENV", "development").lower()
    config_dir = Path(__file__).parent.parent / "config"
    
    base_config = _load_config_file(config_dir / "base.yml")
    env_config_path = config_dir / f"{env}.yml"
    env_config = _load_config_file(env_config_path) if env_config_path.exists() else {}

    def deep_merge(source, destination):
        for key, value in source.items():
            if isinstance(value, dict) and key in destination and isinstance(destination[key], dict):
                destination[key] = deep_merge(value, destination[key])
            else:
                destination[key] = value
        return destination

    merged_config = deep_merge(env_config, base_config)
    log_file_path = Path(merged_config["logging"]["log_file"])
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    processed_data_path = Path(merged_config["data_paths"]["processed_docs"])
    processed_data_path.parent.mkdir(parents=True, exist_ok=True)

    return Settings(**merged_config)

settings = get_settings()