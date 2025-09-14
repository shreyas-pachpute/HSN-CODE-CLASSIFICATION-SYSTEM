import spacy
import re
from typing import Dict, Any, List, Tuple

from src.rag_system import HSNRAGSystem
from src.conversation_manager import ConversationState
from src.logger_setup import setup_logging
from src.performance_monitor import performance_metric

logger = setup_logging(__name__)

class HSNQueryProcessor:
    def __init__(self, rag_system: HSNRAGSystem):
        self.rag_system = rag_system
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.error("spaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
            raise
        
        self.disambiguation_threshold = 0.15
        self.relevance_threshold = 0.40
        self.hsn_code_re = re.compile(r'\b(\d{8})\b')

    @performance_metric
    async def process_query(self, query: str, state: ConversationState) -> Dict[str, Any]:
        parsed_query = self._parse_natural_language(query)
        
        if state.get_context("disambiguation_options") and parsed_query["intent"] == "selection":
            response = self._handle_disambiguation_selection(parsed_query, state)
            state.add_turn(query, response)
            return response

        state.clear_context()

        if parsed_query["intent"] == "direct_lookup":
            hsn_code = parsed_query["hsn_code"]
            response = self._handle_direct_lookup(hsn_code)
            state.add_turn(query, response)
            return response

        if parsed_query["intent"] == "summarization":
            response = self._handle_summarization_query(query)
            state.add_turn(query, response)
            return response

        retrieved_docs = await self.rag_system.retrieve_documents(query)
        
        top_score = retrieved_docs[0].get("score", 0) if retrieved_docs else 0
        if top_score < self.relevance_threshold:
            response = {
                "type": "no_result",
                "summary": "I'm sorry, but I couldn't find any relevant HSN codes for your query in my knowledge base.",
                "confidence": "Very Low"
            }
            state.add_turn(query, response)
            return response

        is_ambiguous, options = self._identify_ambiguous_cases(retrieved_docs)
        
        if is_ambiguous:
            response = self._generate_disambiguation_prompt(options)
            state.set_context("disambiguation_options", options)
        else:
            response = await self.rag_system.generate_from_docs(query, retrieved_docs)
            response["type"] = "classification_result"
        
        state.add_turn(query, response)
        return response

    def _parse_natural_language(self, query: str) -> Dict[str, Any]:
        doc = self.nlp(query.lower())

        hsn_match = self.hsn_code_re.search(query)
        if hsn_match:
            return {"intent": "direct_lookup", "hsn_code": hsn_match.group(1)}

        intent = "classification"
        selection_keywords = {"select", "choose", "option", "first", "second", "third"}
        summary_keywords = {"overview", "category", "type", "kind", "classification"}

        if any(token.lemma_ in selection_keywords for token in doc) or query.strip().isdigit():
            intent = "selection"
        elif any(token.lemma_ in summary_keywords for token in doc):
            intent = "summarization"
        logger.debug(f"Query '{query}' parsed with intent: {intent}")
        return {"text": query, "intent": intent}


    def _handle_direct_lookup(self, hsn_code: str) -> Dict[str, Any]:
        source_docs = self.rag_system.vector_store.collection.get(ids=[f"hsn_{hsn_code}"], include=["metadatas"])
        
        if not source_docs or not source_docs['metadatas'] or not source_docs['metadatas'][0]:
            return {"type": "no_result", "summary": f"HSN Code {hsn_code} was not found in our database."}

        meta = source_docs['metadatas'][0]
        summary = (
            f"**Information for HSN Code {hsn_code}:**\n\n"
            f"- **Description:** {meta.get('item_description', 'N/A')}\n"
            f"- **Trade Status:** Free\n\n"
            f"**Hierarchy:**\n"
            f"- **Chapter ({meta.get('chapter')}):** {meta.get('chapter_description', 'N/A')}\n"
            f"- **Heading ({meta.get('heading')}):** {meta.get('heading_description', 'N/A')}\n"
            f"- **Subheading ({meta.get('subheading')}):** {meta.get('subheading_description', 'N/A')}"
        )
        return {"type": "classification_result", "summary": summary, "top_matches": [{"hsn_code": hsn_code, "metadata": meta}]}

    def _handle_summarization_query(self, query: str) -> Dict[str, Any]:
        summary = (
            "Chapter 40 covers 'Rubber and Articles Thereof'. This includes a wide range of products from raw materials to finished goods.\n\n"
            "To help me find the correct code, could you specify the product? For example, are you looking for:\n"
            "- Raw materials like **natural rubber latex**?\n"
            "- Intermediate products like **vulcanised rubber sheets**?\n"
            "- Finished articles like **rubber tyres** or **conveyor belts**?"
        )
        return {"type": "clarification_prompt", "summary": summary}

    def _identify_ambiguous_cases(self, retrieved_docs: List[Dict[str, Any]]) -> Tuple[bool, List[Dict[str, Any]]]:
        if len(retrieved_docs) < 2: 
            return False, []
        
        score_first = retrieved_docs[0].get("score", 0)
        score_second = retrieved_docs[1].get("score", 0)
        
        logger.info(f"Ambiguity detected. Scores: {score_first:.4f} vs {score_second:.4f}")
        
        if (score_first - score_second) < self.disambiguation_threshold:
            return True, retrieved_docs[:3]
        return False, []

    def _generate_disambiguation_prompt(self, options: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt_text = "I found a few possible matches. To give you the most accurate HSN code, please help me clarify:\n\n"
        for i, option in enumerate(options, 1):
            metadata = option.get('metadata', {})
            hsn = metadata.get('hsn_code', 'N/A')
            desc = metadata.get('item_description', 'N/A')
            context = option.get('graph_context', 'No additional context.')
            prompt_text += (
                f"**Option {i}: HSN Code {hsn}**\n"
                f"- Description: {desc}\n"
                f"- Context: This code is for products under the category of '{context}'.\n\n"
            )
        prompt_text += "Which option best describes your product? Please enter the option number (e.g., '1')."
        return {"type": "disambiguation", "summary": prompt_text, "options": options}

    def _handle_disambiguation_selection(self, parsed_query: Dict[str, Any], state: ConversationState) -> Dict[str, Any]:
        options = state.get_context("disambiguation_options")
        selection = -1
        try:
            selection = int(re.findall(r'\d+', parsed_query["text"])[0])
        except (IndexError, ValueError):
            return {"summary": "I'm sorry, I didn't understand that selection."}

        if 1 <= selection <= len(options):
            selected_option = options[selection - 1]
            hsn_code = selected_option.get('metadata', {}).get('hsn_code', 'N/A')
            final_response = {
                "type": "classification_result",
                "summary": f"Thank you for clarifying. Based on your selection, the correct classification is HSN Code {hsn_code}.",
                "top_matches": [{"hsn_code": hsn_code, "metadata": selected_option.get('metadata', {})}],
                "confidence": "Very High (User Confirmed)",
                "trade_policy": "Free"
            }
            state.clear_context()
            return final_response
        else:
            return {"summary": "That's not a valid option number."}