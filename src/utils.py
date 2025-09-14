from typing import Dict, Any
from pybreaker import CircuitBreaker
from src.config_loader import settings
from src.logger_setup import setup_logging

logger = setup_logging(__name__)

class CircuitBreakerManager:
    """Manages circuit breakers for external services."""
    llm_breaker = CircuitBreaker(
        fail_max=settings.rag_system.circuit_breaker.fail_max,
        reset_timeout=settings.rag_system.circuit_breaker.reset_timeout,
    )

class MetricsCollector:
    """A simple singleton-like class to collect performance metrics."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MetricsCollector, cls).__new__(cls)
            cls._instance.metrics = {}
        return cls._instance

    def record(self, key: str, value: Any):
        self.metrics[key] = value
        logger.debug(f"Metric recorded: {key} = {value}")

    def get_all(self) -> Dict[str, Any]:
        return self.metrics

metrics_collector = MetricsCollector()