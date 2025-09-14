import sys
from src.config_loader import settings
from src.data_processor import HSNDataProcessor
from src.logger_setup import setup_logging

logger = setup_logging(__name__)

def run_pipeline():
    """
    Executes the full HSN data processing pipeline.
    """
    logger.info("Starting HSN Data Processing Pipeline...")
    logger.info(f"Running in '{settings.logging.level}' mode.")

    try:
        processor = HSNDataProcessor(file_path=str(settings.data_paths.raw_hsn_data))
        processor.load_hsn_dataset()
        if not processor.validate_data_quality():
            logger.error("Data quality checks failed. Aborting pipeline.")
            sys.exit(1) 
        processor.enhance_hierarchy()
        structured_documents = processor.create_structured_documents()
        processor.save_documents(
            documents=structured_documents,
            output_path=str(settings.data_paths.processed_docs)
        )

        logger.info("HSN Data Processing Pipeline completed successfully.")

    except FileNotFoundError as e:
        logger.critical(f"A critical file was not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unhandled exception occurred in the pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()