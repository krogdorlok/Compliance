import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib

def train_intent_model(data_path="compliance_chatbot/data/intents.csv", model_path="compliance_chatbot/data/intent_model.pkl"):
    """
    Trains an intent classification model and saves it to a file.
    """
    # Load the data
    df = pd.read_csv(data_path)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["intent"], test_size=0.2, random_state=42
    )

    # Create a pipeline with a TfidfVectorizer and a LogisticRegression classifier
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(random_state=42)),
        ]
    )

    # Train the model
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    accuracy = pipeline.score(X_test, y_test)
    print(f"Intent model accuracy: {accuracy}")

    # Save the model
    joblib.dump(pipeline, model_path)
    print(f"Intent model saved to {model_path}")

if __name__ == "__main__":
    train_intent_model()
