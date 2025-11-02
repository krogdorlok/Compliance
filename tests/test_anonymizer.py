import unittest
from ..app.anonymizer import anonymize_text

class TestAnonymizer(unittest.TestCase):
    def test_anonymize_text(self):
        text = "My name is John Doe and I want to pay my $500 premium."
        anonymized_text = anonymize_text(text)
        self.assertNotIn("John Doe", anonymized_text)
        self.assertIn("[REDACTED_PERSON]", anonymized_text)
        self.assertNotIn("$500", anonymized_text)
        self.assertIn("[REDACTED_AMOUNT]", anonymized_text)

if __name__ == "__main__":
    unittest.main()
