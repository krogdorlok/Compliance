import datetime
from typing import Optional, Dict, Any

from sqlmodel import Field, SQLModel, JSON, Column


class User(SQLModel, table=True):
    """
    Represents a user in the database.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow, nullable=False)


class ChatLog(SQLModel, table=True):
    """
    Represents a chat log entry in the database.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    user_query: str
    anonymized_query: str
    intent: str
    entities: Dict[str, Any] = Field(sa_column=Column(JSON))
    response: str
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow, nullable=False)
