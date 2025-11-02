import csv
import random
from faker import Faker

fake = Faker()

def generate_intents_data(filename="compliance_chatbot/data/intents.csv", num_rows=1000):
    """
    Generates sample data for intent classification.
    """
    intents = ["renewal", "claim", "payment", "quote", "complaint"]
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["text", "intent"])
        for _ in range(num_rows):
            intent = random.choice(intents)
            text = ""
            if intent == "renewal":
                text = f"I want to renew my {fake.word()} policy."
            elif intent == "claim":
                text = f"I need to file a claim for my {fake.word()} policy."
            elif intent == "payment":
                text = f"How can I make a payment for my {fake.word()} policy?"
            elif intent == "quote":
                text = f"Can I get a quote for a {fake.word()} policy?"
            elif intent == "complaint":
                text = f"I have a complaint about my {fake.word()} policy."
            writer.writerow([text, intent])

def generate_ner_data(filename="compliance_chatbot/data/ner_examples.csv", num_rows=500):
    """
    Generates sample data for named entity recognition.
    """
    policy_types = ["auto", "home", "life", "health"]
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["text", "policy_type", "premium_amount", "coverage"])
        for _ in range(num_rows):
            policy_type = random.choice(policy_types)
            premium_amount = round(random.uniform(100, 5000), 2)
            coverage = round(random.uniform(10000, 1000000), 2)
            text = f"My {policy_type} insurance has a premium of ${premium_amount} and coverage of ${coverage}."
            writer.writerow([text, policy_type, premium_amount, coverage])

if __name__ == "__main__":
    generate_intents_data()
    generate_ner_data()
    print("Sample data generated successfully.")
