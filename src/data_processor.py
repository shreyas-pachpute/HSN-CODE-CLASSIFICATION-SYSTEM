import pandas as pd
import pandera as pa
from pandera.errors import SchemaError
from typing import List, Dict, Any
from src.config_loader import settings
from src.logger_setup import setup_logging
from src.performance_monitor import performance_metric

logger = setup_logging(__name__)

class HSNDataProcessor:
    """
    A class to process HSN code data for a RAG system.
    It handles loading, validation, hierarchical enhancement, and structuring
    of HSN data into documents suitable for embedding.
    """

    def __init__(self, file_path: str):
        """
        Initializes the HSNDataProcessor.

        Args:
            file_path (str): The path to the raw HSN JSON dataset.
        
        Raises:
            FileNotFoundError: If the specified file_path does not exist.
        """
        self.file_path = file_path
        self.df = None
        self.hierarchy_map = {}
        self._validate_file_exists()
        self._define_schema()

    def _validate_file_exists(self):
        """Checks if the input data file exists."""
        import os
        if not os.path.exists(self.file_path):
            logger.error(f"Data file not found at path: {self.file_path}")
            raise FileNotFoundError(f"Data file not found: {self.file_path}")

    def _define_schema(self):
        """
        Defines the Pandera schema for input data validation.
        This centralizes data quality rules.
        """
        self.schema = pa.DataFrameSchema({
            "ChapterNumber": pa.Column(int, checks=pa.Check.in_range(
                settings.data_validation.min_chapter_number,
                settings.data_validation.max_chapter_number
            )),
            "HSN Code": pa.Column(str, checks=[
                pa.Check.str_length(
                    settings.data_validation.hsn_code_length,
                    settings.data_validation.hsn_code_length
                ),
                pa.Check.str_matches(r'^\d+$')
            ]),
            "Description": pa.Column(str, nullable=False),
            "Chapter_Description": pa.Column(str, nullable=False),
            "Heading_Description": pa.Column(str, nullable=True),
            "Subheading_Description": pa.Column(str, nullable=True),
        }, strict=False, ordered=False)

    @performance_metric
    def _clean_and_impute_data(self):
        """
        Cleans and imputes missing data.
        Specifically, it forward-fills missing hierarchical descriptions.
        This assumes the data is sorted and a missing heading/subheading
        implies it belongs to the last one seen.
        """
        if self.df is None:
            return

        logger.info("Cleaning and imputing missing hierarchical descriptions...")
        self.df.sort_values(by="HSN Code", inplace=True)
        self.df['Heading_Description'] = self.df.groupby('ChapterNumber')['Heading_Description'].ffill()
        self.df['Subheading_Description'] = self.df.groupby('ChapterNumber')['Subheading_Description'].ffill()
        self.df.fillna({
            'Heading_Description': 'Not specified',
            'Subheading_Description': 'Not specified'
        }, inplace=True)
        logger.info("Data cleaning complete.")


    @performance_metric
    def load_hsn_dataset(self) -> 'HSNDataProcessor':
        """
        Loads the HSN dataset, cleans it, and validates its structure.

        Returns:
            HSNDataProcessor: The instance of the class for method chaining.
        
        Raises:
            ValueError: If the file is empty or cannot be parsed as JSON.
            SchemaError: If the loaded data fails validation against the defined schema.
        """
        logger.info(f"Loading HSN dataset from {self.file_path}...")
        try:
            self.df = pd.read_json(self.file_path)
            if self.df.empty:
                raise ValueError("Loaded data is empty.")
            
            self.df['HSN Code'] = self.df['HSN Code'].astype(str).str.zfill(
                settings.data_validation.hsn_code_length
            )
            
            self._clean_and_impute_data()

            logger.info("Validating data schema...")
            self.schema.validate(self.df, lazy=True)
            logger.info("Data schema validation successful.")
            
        except ValueError as e:
            logger.error(f"Error reading or parsing JSON file: {e}")
            raise
        except SchemaError as e:
            logger.error(f"Data validation failed: {e.errors}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during data loading: {e}", exc_info=True)
            raise
        
        logger.info(f"Successfully loaded, cleaned, and validated {len(self.df)} records.")
        return self

    @performance_metric
    def enhance_hierarchy(self) -> 'HSNDataProcessor':
        """
        Creates a complete hierarchical mapping from the dataset.
        This map connects 2-digit chapters to 4-digit headings and 6-digit subheadings.
        This pre-computation is crucial for efficiently building document context.

        Returns:
            HSNDataProcessor: The instance of the class for method chaining.
        """
        if self.df is None:
            raise RuntimeError("Dataset not loaded. Please call load_hsn_dataset() first.")
        
        logger.info("Building HSN hierarchy map...")
        for _, row in self.df.iterrows():
            hsn_code = row['HSN Code']
            chapter = hsn_code[:2]
            heading = hsn_code[:4]
            subheading = hsn_code[:6]

            if chapter not in self.hierarchy_map:
                self.hierarchy_map[chapter] = {
                    'description': row['Chapter_Description'],
                    'children': {}
                }
            if heading not in self.hierarchy_map[chapter]['children']:
                self.hierarchy_map[chapter]['children'][heading] = {
                    'description': row['Heading_Description'],
                    'children': {}
                }
            if subheading not in self.hierarchy_map[chapter]['children'][heading]['children']:
                 self.hierarchy_map[chapter]['children'][heading]['children'][subheading] = {
                    'description': row['Subheading_Description']
                }
        logger.info("HSN hierarchy map built successfully.")
        return self

    @performance_metric
    def create_structured_documents(self) -> List[Dict[str, Any]]:
        """
        Generates enriched documents for RAG ingestion.
        Each document contains the full hierarchical context in both the text
        and the metadata for downstream use by the Knowledge Graph.
        """
        if self.df is None:
            raise RuntimeError("Dataset not loaded. Please call load_hsn_dataset() first.")
        if not self.hierarchy_map:
            logger.warning("Hierarchy map is empty. Call enhance_hierarchy() for richer documents.")

        logger.info("Creating structured documents for RAG...")
        documents = []
        for _, row in self.df.iterrows():
            hsn_code = row['HSN Code']
            chapter, heading, subheading = hsn_code[:2], hsn_code[:4], hsn_code[:6]

            chapter_desc = self.hierarchy_map.get(chapter, {}).get('description', 'N/A')
            heading_desc = self.hierarchy_map.get(chapter, {}).get('children', {}).get(heading, {}).get('description', 'N/A')
            subheading_desc = self.hierarchy_map.get(chapter, {}).get('children', {}).get(heading, {}).get('children', {}).get(subheading, {}).get('description', 'N/A')
            text_content = (
                f"Product: {row['Description']}. "
                f"Category: {subheading_desc}. "
                f"Broader Group: {heading_desc}. "
                f"General Chapter: {chapter_desc}. "
                f"HSN Code is {hsn_code}."
            )
            
            document = {
                "document_id": f"hsn_{hsn_code}",
                "text": text_content,
                "metadata": {
                    "hsn_code": hsn_code,
                    "chapter": chapter,
                    "heading": heading,
                    "subheading": subheading,
                    "item_description": row['Description'],
                    "chapter_description": chapter_desc,
                    "heading_description": heading_desc,
                    "subheading_description": subheading_desc,
                    "source": self.file_path
                }
            }
            documents.append(document)
        
        logger.info(f"Created {len(documents)} structured documents with optimized text.")
        return documents

    @performance_metric
    def validate_data_quality(self) -> bool:
        """
        Performs comprehensive data quality checks beyond schema validation.

        Checks for:
        - Missing values in key descriptive columns.
        - Logical consistency in the HSN code hierarchy.

        Returns:
            bool: True if all quality checks pass, False otherwise.
        """
        if self.df is None:
            raise RuntimeError("Dataset not loaded. Please call load_hsn_dataset() first.")
        
        logger.info("Performing final data quality checks...")
        quality_ok = True

        critical_cols = ['Description', 'Chapter_Description', 'Heading_Description', 'Subheading_Description']
        if self.df[critical_cols].isnull().values.any():
            logger.warning("Missing values found in critical description columns.")
            quality_ok = False
        
        for _, row in self.df.iterrows():
            hsn_str = row['HSN Code']
            if not (hsn_str[2:4] != '00' or hsn_str[4:6] != '0000'):
                 continue
            
            parent_heading_desc = self.df[self.df['HSN Code'].str.startswith(hsn_str[:4])]['Heading_Description'].iloc[0]
            if not parent_heading_desc or pd.isna(parent_heading_desc):
                logger.warning(f"Inconsistent hierarchy: HSN {hsn_str} has no parent heading description.")
                quality_ok = False

        if quality_ok:
            logger.info("All data quality checks passed.")
        else:
            logger.warning("One or more data quality checks failed.")
            
        return quality_ok

    def save_documents(self, documents: List[Dict[str, Any]], output_path: str):
        """
        Saves the structured documents to a JSON file.

        Args:
            documents (List[Dict[str, Any]]): The list of documents to save.
            output_path (str): The path to the output JSON file.
        """
        import json
        logger.info(f"Saving {len(documents)} documents to {output_path}...")
        try:
            with open(output_path, 'w') as f:
                json.dump(documents, f, indent=4)
            logger.info("Documents saved successfully.")
        except IOError as e:
            logger.error(f"Failed to write documents to file: {e}")
            raise