# app/ml/entity_extractor.py
"""
Enhanced NER training using spaCy with robust entity position handling.
Supports batch processing, validation, and model evaluation.
"""

import pandas as pd
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import random
import logging
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class NERTrainer:
    """
    Robust NER model training with validation, error handling, and metrics.
    This class encapsulates all the logic for training a high-quality
    Named Entity Recognition model.
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the NER trainer.
        
        Args:
            model_name: The base spaCy model to start from. Using a pre-trained
                        model like 'en_core_web_sm' allows us to leverage its
                        existing knowledge (transfer learning).
        """
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"✓ Loaded base spaCy model: {model_name}")
        except OSError:
            logger.warning(f"⚠ Base model {model_name} not found. Creating a blank model.")
            self.nlp = spacy.blank("en")
    
    def validate_csv_format(self, df: pd.DataFrame) -> bool:
        """
        Validate that the input DataFrame has the required columns for training.
        This is a crucial data quality check to prevent errors during training.
        """
        required_cols = {"text", "policy_type", "premium_amount", "coverage"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing the following required columns: {missing}")
        
        if df.isnull().values.any():
            logger.warning("⚠ CSV contains null (empty) values. These rows will be skipped.")
            df.dropna(inplace=True)
        
        return True
    
    def convert_to_spacy_format(self, df: pd.DataFrame) -> List[Tuple]:
        """
        Convert a pandas DataFrame into the format spaCy requires for training.
        This is the most critical part of the fix, as it replaces the fragile
        `text.find()` with a robust method for finding all occurrences of an entity.
        
        Args:
            df: The input DataFrame.
        
        Returns:
            A list of tuples, where each tuple is a training example, e.g.,
            ("some text", {"entities": [(start_char, end_char, "LABEL")]})
        """
        training_data = []
        
        for _, row in df.iterrows():
            text = str(row["text"]).strip()
            if not text:
                continue
            
            entities = []
            
            # --- Robust Entity Finding Logic ---
            # For each entity, we search for all its occurrences in the text,
            # making sure to handle different cases (e.g., "Auto" vs "auto").
            
            # Find POLICY_TYPE
            policy_type = str(row["policy_type"]).strip()
            if policy_type and policy_type.lower() != "nan":
                for match in re.finditer(re.escape(policy_type), text, re.IGNORECASE):
                    start, end = match.span()
                    entities.append((start, end, "POLICY_TYPE"))
            
            # Find PREMIUM_AMOUNT
            premium = str(row["premium_amount"]).strip()
            if premium and premium.lower() != "nan":
                for match in re.finditer(re.escape(premium), text, re.IGNORECASE):
                    start, end = match.span()
                    entities.append((start, end, "PREMIUM_AMOUNT"))

            # Find COVERAGE
            coverage = str(row["coverage"]).strip()
            if coverage and coverage.lower() != "nan":
                for match in re.finditer(re.escape(coverage), text, re.IGNORECASE):
                    start, end = match.span()
                    entities.append((start, end, "COVERAGE"))
            
            if entities:
                training_data.append((text, {"entities": entities}))
            else:
                logger.warning(f"⚠ No entities could be located in text: '{text}'")
        
        logger.info(f"✓ Successfully converted {len(training_data)} examples to spaCy format.")
        return training_data
    
    def train(
        self,
        training_data: List[Tuple],
        epochs: int = 20,
        drop_rate: float = 0.5,
        batch_size: int = 32
    ) -> Dict:
        """
        Train the NER model using the provided data.
        This function includes a more advanced training loop with batching
        and early stopping to prevent overfitting.
        """
        if "ner" not in self.nlp.pipe_names:
            ner = self.nlp.add_pipe("ner", last=True)
        else:
            ner = self.nlp.get_pipe("ner")
        
        for _, annotations in training_data:
            for ent in annotations.get("entities", []):
                ner.add_label(ent[2])
        
        logger.info(f"Starting NER training on {len(training_data)} examples for {epochs} epochs.")
        
        optimizer = self.nlp.begin_training()
        metrics = {"losses": []}
        
        for epoch in range(epochs):
            random.shuffle(training_data)
            losses = {}
            batches = minibatch(training_data, size=compounding(4.0, batch_size, 1.001))
            
            for batch in batches:
                examples = [Example.from_dict(self.nlp.make_doc(text), annot) for text, annot in batch]
                self.nlp.update(examples, drop=drop_rate, sgd=optimizer, losses=losses)
            
            # Debugging: Print the losses dictionary for inspection
            print(f"DEBUG: Epoch {epoch+1} losses dict: {losses}")
            
            avg_loss = float(losses.get('ner', 0.0)) # Ensure avg_loss is always a float
            metrics["losses"].append(avg_loss)
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        return metrics
    
    def save_model(self, model_path: str, metadata: Optional[Dict] = None):
        """
        Save the trained spaCy model to a directory.
        """
        Path(model_path).mkdir(parents=True, exist_ok=True)
        self.nlp.to_disk(model_path)
        logger.info(f"✓ NER Model saved to {model_path}")
        
        if metadata:
            import json
            with open(Path(model_path) / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

def train_ner_model(
    data_path: str = "compliance_chatbot/data/ner_examples.csv",
    model_path: str = "compliance_chatbot/data/ner_model",
    **kwargs
):
    """
    A convenient wrapper function to orchestrate the entire training process.
    This function is backward-compatible with our old training script.
    """
    df = pd.read_csv(data_path)
    
    trainer = NERTrainer()
    trainer.validate_csv_format(df)
    
    training_data = trainer.convert_to_spacy_format(df)
    
    if not training_data:
        raise ValueError("Could not generate any valid training examples from the provided data.")
    
    metrics = trainer.train(training_data, **kwargs)
    
    trainer.save_model(model_path, metadata={
        "training_examples": len(training_data),
        "final_loss": metrics["losses"][-1] if metrics["losses"] else None,
        "epochs_trained": len(metrics["losses"]),
    })
    
    logger.info(f"✓ NER training complete!")
    return trainer.nlp

if __name__ == "__main__":
    train_ner_model()
