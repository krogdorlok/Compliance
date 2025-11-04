import joblib
import spacy
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlmodel import SQLModel, Session

from . import models
from .anonymizer import anonymize_text
from .db import crud
from .db.database import engine, get_db
from .response_generator import ResponseGenerator # Import the new ResponseGenerator

intent_model = None
ner_model = None
response_generator = None # Add a global variable for the response generator


def create_db_and_tables():
    """
    Creates the database and tables if they don't exist.
    """
    SQLModel.metadata.create_all(engine)


app = FastAPI()


class ChatRequest(BaseModel):
    user_id: str
    query: str


@app.on_event("startup")
def on_startup():
    """
    Event handler for application startup.
    """
    global intent_model, ner_model, response_generator
    create_db_and_tables()
    intent_model = joblib.load("compliance_chatbot/data/intent_model.pkl")
    ner_model = spacy.load("compliance_chatbot/data/ner_model")
    response_generator = ResponseGenerator() # Initialize the response generator


@app.get("/health")
def health_check():
    """
    Health check endpoint.
    """
    return {"status": "ok"}


@app.post("/predict_intent")
def predict_intent(request: ChatRequest):
    """
    Predicts the intent of a user's query.
    """
    prediction = intent_model.predict([request.query])
    return {"intent": prediction[0]}


@app.post("/extract_entities")
def extract_entities(request: ChatRequest):
    """
    Extracts entities from a user's query.
    """
    doc = ner_model(request.query)
    entities = {ent.label_: ent.text for ent in doc.ents}
    return entities


@app.post("/chat")
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Handles a user's chat query.
    """
    # Get or create the user
    user = crud.get_user_by_username(db, username=request.user_id)
    if not user:
        user = crud.create_user(db, username=request.user_id)

    # Predict intent
    intent = intent_model.predict([request.query])[0]

    # Extract entities
    doc = ner_model(request.query)
    entities = {ent.label_: ent.text for ent in doc.ents}

    # Anonymize the query
    anonymized_query, audit_log = anonymize_text(request.query) # Anonymizer now returns audit log

    # Generate a response using the ResponseGenerator
    response, response_metadata = response_generator.generate_response(
        intent=intent,
        entities=entities,
        confidence=1.0 # Placeholder, actual confidence would come from intent model
    )

    # Create a chat log
    crud.create_chat_log(
        db=db,
        user_id=user.id,
        user_query=request.query,
        anonymized_query=anonymized_query,
        intent=intent,
        entities=entities,
        response=response,
    )

    return {
        "response": response,
        "intent": intent,
        "entities": entities,
        "anonymized_record_saved": True,
    }
