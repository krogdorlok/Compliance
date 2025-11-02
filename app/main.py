import joblib
import spacy
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlmodel import SQLModel, Session

from . import models
from .anonymizer import anonymize_text
from .db import crud
from .db.database import engine, get_db

intent_model = None
ner_model = None


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
    global intent_model, ner_model
    create_db_and_tables()
    intent_model = joblib.load("compliance_chatbot/data/intent_model.pkl")
    ner_model = spacy.load("compliance_chatbot/data/ner_model")


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
    anonymized_query = anonymize_text(request.query)

    # Generate a response (simple logic for now)
    response = f"Intent: {intent}, Entities: {entities}"

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
