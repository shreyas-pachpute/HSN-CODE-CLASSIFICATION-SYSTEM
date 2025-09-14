from typing import List, Dict, Any, Optional
from uuid import uuid4

class ConversationState:
    """
    Manages the state for a single user conversation session.

    This class tracks the history of queries, responses, and any context
    needed for follow-up questions, such as disambiguation options.
    """
    def __init__(self, session_id: Optional[str] = None):
        self.session_id: str = session_id or str(uuid4())
        self.turn_history: List[Dict[str, Any]] = []
        self.current_context: Dict[str, Any] = {}
        self.user_preferences: Dict[str, Any] = {"expertise_level": "novice"}

    def add_turn(self, user_query: str, system_response: Dict[str, Any]):
        """Adds a user query and system response to the history."""
        self.turn_history.append({
            "user_query": user_query,
            "system_response": system_response
        })

    def set_context(self, context_key: str, value: Any):
        """Sets a specific piece of context, e.g., for disambiguation."""
        self.current_context[context_key] = value

    def get_context(self, context_key: str) -> Optional[Any]:
        """Retrieves a piece of context."""
        return self.current_context.get(context_key)

    def clear_context(self):
        """Clears the context after it's been used."""
        self.current_context = {}

    def get_full_history_str(self) -> str:
        """Returns the conversation history as a formatted string."""
        history_str = ""
        for turn in self.turn_history:
            history_str += f"User: {turn['user_query']}\n"
            summary = turn['system_response'].get('summary', 'No summary.')
            history_str += f"System: {summary}\n"
        return history_str