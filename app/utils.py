import csv

def append_to_csv(filename: str, data: list):
    """
    Appends a new row of data to a CSV file.

    Args:
        filename: The path to the CSV file.
        data: A list of values to be added as a new row.
    """
    with open(filename, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)

def add_intent_example(text: str, intent: str, filename: str = "compliance_chatbot/data/intents.csv"):
    """
    Adds a new example to the intent classification dataset.

    Args:
        text: The user query.
        intent: The corresponding intent.
        filename: The path to the intents CSV file.
    """
    append_to_csv(filename, [text, intent])

def add_ner_example(text: str, policy_type: str, premium_amount: float, coverage: float, filename: str = "compliance_chatbot/data/ner_examples.csv"):
    """
    Adds a new example to the named entity recognition dataset.

    Args:
        text: The user query.
        policy_type: The type of policy.
        premium_amount: The premium amount.
        coverage: The coverage amount.
        filename: The path to the NER examples CSV file.
    """
    append_to_csv(filename, [text, policy_type, premium_amount, coverage])
