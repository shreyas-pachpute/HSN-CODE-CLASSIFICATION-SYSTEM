import pytest
from src.conversation_manager import ConversationState
import string

pytestmark = pytest.mark.asyncio

async def test_full_system_scenarios(initialized_query_processor, test_scenarios):
    """
    Executes the entire suite of test scenarios against the live system.
    """
    for scenario in test_scenarios:
        print(f"\n--- Running Test: {scenario['id']} - {scenario['description']} ---")
        
        query = scenario["query"]
        expected = scenario["expected"]
        
        state = ConversationState()
        if "context" in scenario:
            for key, value in scenario["context"].items():
                state.set_context(key, value)

        response = await initialized_query_processor.process_query(query, state)

        assert "summary" in response, f"[{scenario['id']}] Response missing 'summary'"
        
        if "type" in expected:
            assert response.get("type") == expected["type"], \
                f"[{scenario['id']}] Expected type '{expected['type']}', got '{response.get('type')}'"

        if "top_hsn_code" in expected:
            assert "top_matches" in response and len(response["top_matches"]) > 0, \
                f"[{scenario['id']}] Response is missing 'top_matches' when one was expected."
            
            # Fix: Check multiple possible structures for metadata access
            first_match = response["top_matches"][0]
            actual_hsn = None
            
            if "hsn_code" in first_match:
                actual_hsn = first_match["hsn_code"]
            elif "metadata" in first_match and "hsn_code" in first_match["metadata"]:
                actual_hsn = first_match["metadata"]["hsn_code"]
            else:
                actual_hsn = first_match.get("metadata", {}).get("hsn_code", "N/A")
                
            assert actual_hsn == expected["top_hsn_code"], \
                f"[{scenario['id']}] Expected top HSN '{expected['top_hsn_code']}', got '{actual_hsn}'"

        if "must_contain" in expected:
            text_to_check = response["summary"]
            if response.get("type") == "disambiguation":
                option_texts = []
                for opt in response.get("options", []):
                    if "metadata" in opt and "hsn_code" in opt["metadata"]:
                        option_texts.append(opt["metadata"]["hsn_code"])
                    elif "hsn_code" in opt:
                        option_texts.append(opt["hsn_code"])
                    elif isinstance(opt.get("metadata"), dict):
                        hsn_code = opt["metadata"].get("hsn_code")
                        if hsn_code:
                            option_texts.append(hsn_code)
                
                text_to_check += " " + " ".join(option_texts)

            summary_lower = text_to_check.lower()
            for text in expected["must_contain"]:
                assert any(clean(text) in clean(line) for line in summary_lower.split('\n')), \
                    f"[{scenario['id']}] Response did not contain expected text: '{text}'"

        if "confidence" in expected:
             assert expected["confidence"].lower() in response.get("confidence", "").lower(), \
                f"[{scenario['id']}] Expected confidence '{expected['confidence']}', got '{response.get('confidence')}'"
                
def clean(s):
    import re
    s = s.lower()
    s = s.replace(":", "").replace("-", "")
    s = re.sub(r"\\*+", "", s)  # Remove asterisks (from markdown bold)
    s = s.strip()
    s = s.rstrip(string.punctuation)
    return s
