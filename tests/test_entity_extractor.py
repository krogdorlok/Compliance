# tests/test_entity_extractor.py
"""
Tests for the enhanced NERTrainer class.
"""

import unittest
import pandas as pd
from app.ml.entity_extractor import NERTrainer

class TestNERTrainer(unittest.TestCase):
    """Test suite for the NER training logic."""

    def setUp(self):
        """Set up a trainer instance before each test."""
        self.trainer = NERTrainer(model_name="en_core_web_sm")
        self.sample_data = {
            "text": [
                "My auto insurance has a premium of $150.50 and coverage of $50000.",
                "I want to know about my home policy.",
                "The coverage for my life plan is 250000."
            ],
            "policy_type": ["auto", "home", "life"],
            "premium_amount": ["150.50", None, None],
            "coverage": ["50000", None, "250000"]
        }
        self.df = pd.DataFrame(self.sample_data)

    def test_validate_csv_format_success(self):
        """Test that a correctly formatted DataFrame passes validation."""
        self.assertTrue(self.trainer.validate_csv_format(self.df))

    def test_validate_csv_format_failure(self):
        """Test that a DataFrame with missing columns fails validation."""
        invalid_df = self.df.drop(columns=["policy_type"])
        with self.assertRaises(ValueError):
            self.trainer.validate_csv_format(invalid_df)

    def test_convert_to_spacy_format(self):
        """Test the conversion of a DataFrame to spaCy's training format."""
        training_data = self.trainer.convert_to_spacy_format(self.df)
        
        # We expect 3 valid examples
        self.assertEqual(len(training_data), 3)
        
        # Check the first, most complex example
        first_example = training_data[0]
        text, annotations = first_example
        entities = annotations["entities"]
        
        self.assertEqual(text, "My auto insurance has a premium of $150.50 and coverage of $50000.")
        
        # Check that all three entities were found
        self.assertEqual(len(entities), 3)
        
        # Verify the labels and positions of the entities
        entity_labels = {label for _, _, label in entities}
        self.assertIn("POLICY_TYPE", entity_labels)
        self.assertIn("PREMIUM_AMOUNT", entity_labels)
        self.assertIn("COVERAGE", entity_labels)

    def test_training_loop_runs(self):
        """Test that the training loop runs without crashing on a small dataset."""
        training_data = self.trainer.convert_to_spacy_format(self.df)
        
        # Run for a small number of epochs to ensure it works
        metrics = self.trainer.train(training_data, epochs=2)
        
        self.assertIn("losses", metrics)
        self.assertEqual(len(metrics["losses"]), 2)
        self.assertTrue(all(isinstance(loss, float) for loss in metrics["losses"]))

if __name__ == "__main__":
    unittest.main()
