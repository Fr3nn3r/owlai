import sys
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


def test_connection():
    # Connection string with SSL verification disabled (NOT recommended for production)
    connection_string = (
        "postgresql://owluser:NyCINUy7Un3JjE28Md3mRjpg5Dd4aKEy@"
        "dpg-cvn7c2ngi27c73bi26hg-a.frankfurt-postgres.render.com/owlai_db"
        "?sslmode=prefer"
    )

    try:
        # Create engine
        engine = create_engine(connection_string)

        print("Attempting to connect to the database...")

        # Test connection
        with engine.connect() as connection:
            result = connection.execute(text("SELECT version();"))
            version = result.scalar()

            if version:
                print("Successfully connected to the database!")
                print(f"PostgreSQL version: {version}")
            else:
                print("Successfully connected but could not retrieve version.")

            print("Connection closed successfully.")

    except SQLAlchemyError as e:
        print(f"Error connecting to the database: {e}")
        print("Error details:", str(e.__dict__))
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Error details:", str(e.__dict__))
        sys.exit(1)


if __name__ == "__main__":
    test_connection()
