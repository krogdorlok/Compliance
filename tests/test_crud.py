import unittest
from sqlmodel import create_engine, Session, SQLModel
from app.db import crud
from app import models

DATABASE_URL = "sqlite:///test.db"
engine = create_engine(DATABASE_URL, echo=True)

class TestCrud(unittest.TestCase):
    def setUp(self):
        SQLModel.metadata.create_all(engine)
        self.db = Session(engine)

    def tearDown(self):
        SQLModel.metadata.drop_all(engine)
        self.db.close()

    def test_create_and_get_user(self):
        user = crud.create_user(self.db, "testuser")
        self.assertEqual(user.username, "testuser")
        retrieved_user = crud.get_user_by_username(self.db, "testuser")
        self.assertEqual(retrieved_user.id, user.id)

    def test_create_chat_log(self):
        user = crud.create_user(self.db, "testuser")
        chat_log = crud.create_chat_log(
            db=self.db,
            user_id=user.id,
            user_query="Hello",
            anonymized_query="Hello",
            intent="greeting",
            entities={},
            response="Hi there!",
        )
        self.assertEqual(chat_log.user_query, "Hello")
        self.assertEqual(chat_log.intent, "greeting")

if __name__ == "__main__":
    unittest.main()
