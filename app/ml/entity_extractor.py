import pandas as pd
import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import random

def train_ner_model(data_path="compliance_chatbot/data/ner_examples.csv", model_path="compliance_chatbot/data/ner_model"):
    """
    Trains a named entity recognition model using spaCy and saves it.
    """
    # Load the data
    df = pd.read_csv(data_path)

    # Create a blank spaCy model
    nlp = spacy.blank("en")

    # Create a new entity recognizer and add it to the pipeline
    ner = nlp.add_pipe("ner")

    # Add labels to the NER pipe
    for _, row in df.iterrows():
        ner.add_label("POLICY_TYPE")
        ner.add_label("PREMIUM_AMOUNT")
        ner.add_label("COVERAGE")

    # Prepare the training data
    train_data = []
    for _, row in df.iterrows():
        text = row["text"]
        entities = []
        
        policy_type = str(row["policy_type"])
        start_index = text.find(policy_type)
        if start_index != -1:
            end_index = start_index + len(policy_type)
            entities.append((start_index, end_index, "POLICY_TYPE"))

        premium_amount = str(row["premium_amount"])
        start_index = text.find(premium_amount)
        if start_index != -1:
            end_index = start_index + len(premium_amount)
            entities.append((start_index, end_index, "PREMIUM_AMOUNT"))
            
        coverage = str(row["coverage"])
        start_index = text.find(coverage)
        if start_index != -1:
            end_index = start_index + len(coverage)
            entities.append((start_index, end_index, "COVERAGE"))
            
        train_data.append((text, {"entities": entities}))

    # Train the model
    optimizer = nlp.begin_training()
    for i in range(20):
        random.shuffle(train_data)
        losses = {}
        for text, annotations in train_data:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)
        print(f"Epoch {i+1}, Losses: {losses}")

    # Save the model
    nlp.to_disk(model_path)
    print(f"NER model saved to {model_path}")

if __name__ == "__main__":
    train_ner_model()
