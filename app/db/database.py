from sqlmodel import create_engine, Session

DATABASE_URL = "sqlite:///compliance_chatbot.db"

engine = create_engine(DATABASE_URL, echo=True)

def get_db():
    """
    Returns a database session.
    """
    with Session(engine) as session:
        yield session
