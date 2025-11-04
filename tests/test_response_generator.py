# tests/test_response_generator.py
"""
Tests for the ResponseGenerator class.
"""

import unittest
import json
from pathlib import Path
from app.response_generator import ResponseGenerator

# Define a temporary knowledge base path for testing
TEST_KB_PATH = "compliance_chatbot/data/test_faq_knowledge_base.json"

class TestResponseGenerator(unittest.TestCase):
    """Test suite for the ResponseGenerator functionality."""

    def setUp(self):
        """Set up a temporary knowledge base file for testing."""
        self.test_kb_content = {
            "greeting": {
                "template": "Hello! How can I assist you today?",
                "confidence_threshold": 0.9,
                "fallback": "I'm here to help. What can I do for you?",
                "examples": ["hi", "hello", "good morning"]
            },
            "renewal": {
                "template": "To renew your {policy_type} policy, please visit example.com/renew. Your premium is {premium_amount}.",
                "confidence_threshold": 0.7,
                "fallback": "I can help with renewals. Which policy type?",
                "examples": ["renew my policy", "how to renew"]
            },
            "quote": {
                "template": "To get a quote for a {policy_type} policy with {coverage} coverage, visit example.com/quotes.",
                "confidence_threshold": 0.6,
                "fallback": "I can provide quotes. What type of policy are you interested in?",
                "examples": ["get a quote", "insurance cost"]
            },
            "unmapped_intent": {
                "template": "This is an unmapped template for {entity}.",
                "confidence_threshold": 0.5,
                "fallback": "I don't have a specific answer for this unmapped intent."
            }
        }
        with open(TEST_KB_PATH, "w") as f:
            json.dump(self.test_kb_content, f, indent=2)
        
        self.generator = ResponseGenerator(knowledge_base_path=TEST_KB_PATH)

    def tearDown(self):
        """Clean up the temporary knowledge base file."""
        Path(TEST_KB_PATH).unlink(missing_ok=True)

    def test_load_knowledge_base_success(self):
        """Test successful loading of the knowledge base."""
        self.assertIn("renewal", self.generator.knowledge_base)
        self.assertEqual(len(self.generator.knowledge_base), 4)

    def test_load_knowledge_base_file_not_found(self):
        """Test handling of a missing knowledge base file."""
        # Temporarily remove the test KB
        Path(TEST_KB_PATH).unlink()
        generator = ResponseGenerator(knowledge_base_path="non_existent_kb.json")
        self.assertEqual(generator.knowledge_base, {})

    def test_generate_response_template_match(self):
        """Test generating a response with a matching template and entities."""
        intent = "renewal"
        entities = {"policy_type": "auto", "premium_amount": "$150"}
        response, metadata = self.generator.generate_response(intent, entities)
        self.assertIn("auto policy", response)
        self.assertIn("$150", response)
        self.assertEqual(metadata["source"], "template")
        self.assertEqual(metadata["matched_intent"], "renewal")
        self.assertIn("policy_type", metadata["entities_used"])

    def test_generate_response_fallback_unmapped_intent(self):
        """Test fallback for an intent not in the knowledge base."""
        intent = "unknown_intent"
        entities = {}
        response, metadata = self.generator.generate_response(intent, entities)
        self.assertIn("not sure how to help with that", response)
        self.assertEqual(metadata["source"], "fallback")
        self.assertEqual(metadata["matched_intent"], "unknown_intent")

    def test_generate_response_fallback_low_confidence(self):
        """Test fallback when confidence is below threshold."""
        intent = "renewal"
        entities = {"policy_type": "home"}
        response, metadata = self.generator.generate_response(intent, entities, confidence=0.5) # Below 0.7 threshold
        self.assertIn("I can help with renewals. Which policy type?", response)
        self.assertEqual(metadata["source"], "fallback")

    def test_generate_response_missing_entity_in_template(self):
        """Test fallback when a required entity for the template is missing."""
        intent = "renewal"
        entities = {"premium_amount": "$200"} # Missing policy_type
        response, metadata = self.generator.generate_response(intent, entities)
        self.assertIn("I need more information to answer that. Missing: 'policy_type'.", response)
        self.assertEqual(metadata["source"], "fallback_missing_entity")

    def test_batch_generate_responses(self):
        """Test generating multiple responses in a batch."""
        intents = ["greeting", "renewal", "quote"]
        entities_list = [
            {},
            {"policy_type": "life", "premium_amount": "$300"},
            {"policy_type": "health", "coverage": "100000"}
        ]
        confidences = [0.95, 0.8, 0.65]
        
        results = self.generator.batch_generate(intents, entities_list, confidences)
        self.assertEqual(len(results), 3)
        
        # Verify first response
        self.assertIn("Hello!", results[0][0])
        self.assertEqual(results[0][1]["source"], "template")
        
        # Verify second response
        self.assertIn("life policy", results[1][0])
        self.assertIn("$300", results[1][0])
        self.assertEqual(results[1][1]["source"], "template")

        # Verify third response
        self.assertIn("health policy", results[2][0])
        self.assertIn("100000 coverage", results[2][0])
        self.assertEqual(results[2][1]["source"], "template")

if __name__ == "__main__":
    unittest.main()
