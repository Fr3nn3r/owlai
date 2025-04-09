import logging
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError
from owlai.db.dbmodels import Base, VectorStore
from owlai.services.system import DATABASE_URL

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_vector_store_table():
    logger.info("Starting VectorStore table creation process")

    try:
        # Create engine
        logger.info("Creating database engine")
        engine = create_engine(DATABASE_URL)

        # Check if table already exists
        inspector = inspect(engine)
        if "vector_stores" in inspector.get_table_names():
            logger.warning("VectorStore table already exists! Skipping creation.")
            return False

        # Create only the VectorStore table
        logger.info("Creating VectorStore table")
        Base.metadata.tables["vector_stores"].create(engine)

        logger.info("VectorStore table created successfully!")
        return True

    except SQLAlchemyError as e:
        logger.error(f"Database error occurred: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        create_vector_store_table()
    except Exception as e:
        logger.error("Failed to create VectorStore table")
        raise
