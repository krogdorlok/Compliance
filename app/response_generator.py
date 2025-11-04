# app/response_generator.py
"""
Template-based response generation using intent + entities.
Supports confidence scoring and fallback handling.
"""

import json
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """
    Generates contextual responses based on intent and extracted entities.
    Uses template matching and substitution.
    """
    
    def __init__(self, knowledge_base_path: str = "compliance_chatbot/data/faq_knowledge_base.json"):
        """
        Initialize response generator with FAQ knowledge base.
        
        Args:
            knowledge_base_path: Path to FAQ JSON file
        """
        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
        logger.info(f"✓ Loaded knowledge base with {len(self.knowledge_base)} intents")
    
    def _load_knowledge_base(self, path: str) -> Dict:
        """
        Load FAQ knowledge base from JSON.
        
        Expected format:
        {
          "renewal": {
            "template": "To renew your {policy_type}, please visit...",
            "confidence_threshold": 0.8,
            "examples": ["renew", "renewal", "extend"],
            "fallback": "I can help with renewals. What policy type?"
          }
        }
        """
        if not Path(path).exists():
            logger.warning(f"⚠ Knowledge base not found at {path}, using empty KB")
            return {}
        
        try:
            with open(path, "r") as f:
                kb = json.load(f)
            return kb
        except json.JSONDecodeError as e:
            logger.error(f"✗ Invalid JSON in knowledge base: {e}")
            return {}
    
    def generate_response(
        self,
        intent: str,
        entities: Dict[str, Any],
        confidence: float = 1.0
    ) -> Tuple[str, Dict]:
        """
        Generate a response based on intent and entities.
        
        Args:
            intent: Predicted user intent (e.g., "renewal")
            entities: Extracted entities (e.g., {"policy_type": "term life"})
            confidence: Model confidence in this prediction (0.0-1.0)
        
        Returns:
            Tuple of (response_text, metadata)
            metadata = {
                "source": "template" | "fallback",
                "confidence": float,
                "matched_intent": str,
                "entities_used": list
            }
        """
        metadata = {
            "source": "unknown",
            "confidence": confidence,
            "matched_intent": intent,
            "entities_used": list(entities.keys())
        }
        
        # Check if intent exists in knowledge base
        if intent not in self.knowledge_base:
            metadata["source"] = "fallback"
            response = f"I'm not sure how to help with that. Could you clarify your {intent} request?"
            logger.warning(f"⚠ Intent '{intent}' not in knowledge base")
            return response, metadata
        
        intent_data = self.knowledge_base[intent]
        
        # Check confidence threshold
        threshold = intent_data.get("confidence_threshold", 0.7)
        if confidence < threshold:
            metadata["source"] = "fallback"
            response = intent_data.get("fallback", f"I'm not confident in my understanding. Can you rephrase?")
            logger.info(f"Confidence {confidence} below threshold {threshold}, using fallback")
            return response, metadata
        
        # Generate response from template
        try:
            template = intent_data.get("template", "")
            response = template.format(**entities)
            metadata["source"] = "template"
            logger.info(f"Generated response for intent '{intent}'")
        except KeyError as e:
            # Missing entity in template, prioritize specific error message
            metadata["source"] = "fallback_missing_entity"
            response = f"I need more information to answer that. Missing: '{e.args[0]}'."
            logger.warning(f"Missing entity {e.args[0]} for intent '{intent}', using specific fallback.")
        except Exception as e:
            # Catch any other unexpected errors during template formatting
            metadata["source"] = "fallback_error"
            response = intent_data.get("fallback", "I encountered an unexpected error while generating a response.")
            logger.error(f"Unexpected error during response generation for intent '{intent}': {e}")
        
        return response, metadata
    
    def batch_generate(
        self,
        intents: list,
        entities_list: list,
        confidences: Optional[list] = None
    ) -> list:
        """
        Generate responses for multiple queries (batch processing).
        
        Args:
            intents: List of intents
            entities_list: List of entity dicts
            confidences: Optional list of confidence scores
        
        Returns:
            List of (response, metadata) tuples
        """
        if confidences is None:
            confidences = [1.0] * len(intents)
        
        results = []
        for intent, entities, conf in zip(intents, entities_list, confidences):
            response, meta = self.generate_response(intent, entities, conf)
            results.append((response, meta))
        
        return results
