import time
import functools
from typing import Any, Callable
from src.logger_setup import setup_logging

logger = setup_logging(__name__)

def performance_metric(func: Callable) -> Callable:
    """
    A decorator that logs the execution time of a function.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The wrapped function with timing logic.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Wrapper function to time the execution."""
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            run_time = (end_time - start_time) * 1000 
            logger.info(
                f"Execution of '{func.__name__}' took {run_time:.2f} ms."
            )
    return wrapper