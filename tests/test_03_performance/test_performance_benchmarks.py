import pytest
import asyncio
from src.conversation_manager import ConversationState

def test_benchmark_simple_query(benchmark, sync_initialized_query_processor):
    state = ConversationState()
    query = "Nylon tyre yarm"
    
    def run_async_query():
        return asyncio.run(sync_initialized_query_processor.process_query(query, state))

    result = benchmark(run_async_query)
    
    assert result is not None
    assert result["top_matches"][0]["hsn_code"] == "54021910"

def test_benchmark_ambiguous_query(benchmark, sync_initialized_query_processor):
    state = ConversationState()
    query = "natural rubber latex"
    
    def run_async_query():
        return asyncio.run(sync_initialized_query_processor.process_query(query, state))

    result = benchmark(run_async_query)
    
    assert result is not None
    assert result["type"] == "disambiguation"