# tests/test_anonymizer.py
"""
Comprehensive test suite for anonymizer module.
Tests edge cases, multiple entity types, and batch processing.
"""

import unittest
from app.anonymizer import anonymize_text, batch_anonymize

class TestAnonymizer(unittest.TestCase):
    """Test suite for anonymization functionality."""
    
    def test_anonymize_simple_name(self):
        """Test masking a simple first-last name."""
        text = "My name is John Doe."
        anon_text, log = anonymize_text(text)
        self.assertNotIn("John Doe", anon_text)
        self.assertIn("[REDACTED_PERSON]", anon_text)
        self.assertEqual(log["total_masked"], 1)
    
    def test_anonymize_currency_with_symbol(self):
        """Test masking currency amounts with symbols."""
        text = "The premium is $500."
        anon_text, log = anonymize_text(text)
        self.assertNotIn("$500", anon_text)
        self.assertIn("[REDACTED_AMOUNT]", anon_text)
    
    def test_anonymize_currency_without_symbol(self):
        """Test masking currency amounts without symbols."""
        text = "I paid 500 dollars."
        anon_text, log = anonymize_text(text)
        # Depends on spaCy recognizing "500 dollars" as MONEY
        self.assertTrue(log["total_masked"] > 0)
    
    def test_anonymize_email(self):
        """Test masking email addresses."""
        text = "Contact me at john.doe@example.com."
        anon_text, log = anonymize_text(text)
        self.assertNotIn("john.doe@example.com", anon_text)
        self.assertIn("[REDACTED_EMAIL]", anon_text)
    
    def test_anonymize_phone_number(self):
        """Test masking phone numbers."""
        text = "Call me at 555-123-4567."
        anon_text, log = anonymize_text(text)
        self.assertNotIn("555-123-4567", anon_text)
        self.assertIn("[REDACTED_PHONE]", anon_text)
    
    def test_anonymize_ssn(self):
        """Test masking Social Security Numbers."""
        text = "My SSN is 123-45-6789."
        anon_text, log = anonymize_text(text)
        self.assertNotIn("123-45-6789", anon_text)
        self.assertIn("[REDACTED_SSN]", anon_text)
    
    def test_anonymize_multiple_entities(self):
        """Test masking multiple entity types in one text."""
        text = "John Doe from Acme Corp paid $1,000 on 555-111-2222."
        anon_text, log = anonymize_text(text)
        self.assertGreater(log["total_masked"], 1)
        # Should mask name, org, amount, phone
    
    def test_anonymize_non_ascii(self):
        """Test handling of non-ASCII characters."""
        text = "José García paid €500."
        anon_text, log = anonymize_text(text)
        # Should handle Unicode gracefully
        self.assertIsInstance(anon_text, str)
    
    def test_anonymize_empty_string(self):
        """Test handling of empty input."""
        text = ""
        anon_text, log = anonymize_text(text)
        self.assertEqual(anon_text, "")
        self.assertEqual(log["total_masked"], 0)
    
    def test_anonymize_already_redacted(self):
        """Test that already-redacted text isn't double-redacted."""
        text = "Contact [REDACTED_EMAIL] for support."
        anon_text, log = anonymize_text(text)
        # Should not have double redaction
        self.assertLess(log["total_masked"], 2)
    
    def test_anonymize_audit_log_structure(self):
        """Test that audit log has correct structure."""
        text = "John Doe paid $500."
        anon_text, log = anonymize_text(text)
        self.assertIn("total_masked", log)
        self.assertIn("by_type", log)
        self.assertIn("masked_entities", log)
        self.assertIn("timestamp", log)
    
    def test_batch_anonymize(self):
        """Test batch anonymization of multiple texts."""
        texts = [
            "John Doe paid $500",
            "jane.smith@example.com is the contact",
            "Call 555-222-3333"
        ]
        results = batch_anonymize(texts)
        self.assertEqual(len(results), 3)
        total_masked = sum(log["total_masked"] for _, log in results)
        self.assertGreater(total_masked, 0)
    
    def test_anonymize_selective_entity_types(self):
        """Test anonymizing only specific entity types."""
        text = "John Doe from Acme paid $500."
        # Only mask names, not org or amount
        anon_text, log = anonymize_text(text, include_pii_types=["PERSON", "EMAIL", "PHONE", "SSN"])
        self.assertIn("[REDACTED_PERSON]", anon_text)
        self.assertIn("Acme", anon_text)  # Should NOT be masked
        self.assertIn("$500", anon_text)  # Should NOT be masked

if __name__ == "__main__":
    unittest.main()
