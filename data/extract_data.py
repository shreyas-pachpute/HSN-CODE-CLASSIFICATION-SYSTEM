"""
Extracts HSN entries from the provided PDF using robust table detection.
This version correctly handles multi-line descriptions within table cells.
"""
import json
from pathlib import Path
import pdfplumber
import re
from typing import List, Dict, Any, Optional

# --- Configuration ---
PDF_PATH = Path(r"C:\Users\AAYUSH\Downloads\Trade_Notice_First_50_Pages.pdf")
OUTPUT_PATH = Path(r"D:\SHREYAS\WORKABLEAI\data\raw\hsn_codes.json")

# --- Regular Expressions for parsing cell content ---
# Matches 2, 4, 6, or 8 digit codes that are purely numeric
HSN_CODE_RE = re.compile(r"^\s*(\d{2}|\d{4}|\d{6}|\d{8})\s*$")

def clean_text(text: Optional[str]) -> str:
    """Cleans and normalizes text from a PDF cell."""
    if not text:
        return ""
    # Replace newline characters with spaces and strip whitespace
    return text.replace("\n", " ").strip()

def parse_row(row: List[str], state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Parses a single row from the extracted table to identify HSN entries.
    Updates the state dictionary with the current hierarchy.

    Args:
        row (List[str]): A list of strings representing cells in a table row.
        state (Dict[str, Any]): A dictionary holding the current hierarchical state.

    Returns:
        Optional[Dict[str, Any]]: A dictionary representing an 8-digit HSN item, or None.
    """
    if len(row) < 3:
        return None # Not a valid data row

    chapter_num_str = clean_text(row[0])
    hsn_code_str = clean_text(row[1])
    description = clean_text(row[2])
    export_policy = clean_text(row[3]) if len(row) > 3 else "N/A"

    if not chapter_num_str.isdigit():
        return None # Skip header rows or invalid rows

    match = HSN_CODE_RE.match(hsn_code_str)
    if not match:
        return None # The second column is not a valid HSN code format

    code = match.group(1)
    code_level = len(code)

    # Update state based on the hierarchy level
    if code_level == 2:
        state["chapter_desc"] = description
        state["heading_desc"] = None
        state["subheading_desc"] = None
    elif code_level == 4:
        state["heading_desc"] = description
        state["subheading_desc"] = None
    elif code_level == 6:
        state["subheading_desc"] = description
    elif code_level == 8:
        # This is a final, extractable item
        return {
            "ChapterNumber": int(chapter_num_str),
            "HSN Code": code, # Keep as string to preserve leading zeros
            "Description": description,
            "FinalHSN": export_policy.capitalize(),
            "Chapter_Description": state.get("chapter_desc"),
            "Heading_Description": state.get("heading_desc"),
            "Subheading_Description": state.get("subheading_desc"),
        }
    return None

def main():
    """Main function to extract HSN data from the PDF and save to JSON."""
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

    results = []
    # State dictionary to keep track of the hierarchy across pages
    current_state = {
        "chapter_desc": None,
        "heading_desc": None,
        "subheading_desc": None,
    }

    print(f"Starting extraction from {PDF_PATH}...")
    with pdfplumber.open(PDF_PATH) as pdf:
        for i, page in enumerate(pdf.pages):
            # Skip the first page which is just a notice
            if i == 0:
                continue
            
            print(f"Processing Page {i+1}...")
            # Use pdfplumber's table extraction with a vertical line strategy
            tables = page.extract_tables(table_settings={
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
            })

            if not tables:
                print(f"  - Warning: No tables found on page {i+1}.")
                continue

            for table in tables:
                for row in table:
                    # The first row is often a header, parse_row will skip it
                    extracted_item = parse_row(row, current_state)
                    if extracted_item:
                        results.append(extracted_item)

    print(f"Extraction complete. Found {len(results)} HSN records.")

    # Write the results to the output JSON file
    try:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Successfully saved data to {OUTPUT_PATH.resolve()}")
    except IOError as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    main()