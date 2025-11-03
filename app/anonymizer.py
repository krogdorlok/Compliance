# app/anonymizer.py
"""
Enhanced PII Anonymizer using spaCy NER for robust entity detection.
Supports multiple masking strategies and audit logging.
"""

import spacy
import re
import logging
from typing import Tuple, Dict, List, Optional, Literal
from datetime import datetime

logger = logging.getLogger(__name__)

# Configuration for PII entity types and masking tokens
# This dictionary defines which types of entities (like names, locations)
# should be considered Personally Identifiable Information (PII) and what
# placeholder text should be used to replace them.
PII_ENTITY_CONFIG = {
    "PERSON": {"token": "[REDACTED_PERSON]", "spacy_label": "PERSON"},
    "GPE": {"token": "[REDACTED_LOCATION]", "spacy_label": "GPE"},
    "ORG": {"token": "[REDACTED_ORG]", "spacy_label": "ORG"},
    "MONEY": {"token": "[REDACTED_AMOUNT]", "spacy_label": "MONEY"},
}

# Additional regex-based detections for entities spaCy might miss.
# Regex (Regular Expressions) are powerful patterns for finding text.
# We use them here to catch specific formats like emails and phone numbers
# that spaCy's general model might not be trained on.
ADDITIONAL_PII_PATTERNS = {
    "EMAIL": (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED_EMAIL]'),
    "PHONE": (r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b', '[REDACTED_PHONE]'),
    "SSN": (r'\b(?!000|666)[0-9]{3}-(?!00)[0-9]{2}-(?!0000)[0-9]{4}\b', '[REDACTED_SSN]'),
}

# Global spaCy model (loaded once for efficiency).
# We load the machine learning model into a global variable so that it's
# only loaded into memory once when the application starts, rather than
# every time we need to anonymize a piece of text. This is much faster.
_nlp_model = None

def _load_spacy_model():
    """
    Load spaCy English model with caching.
    This function handles loading the pre-trained 'en_core_web_sm' model.
    If the model isn't downloaded on the system, it logs a warning and
    uses a blank English model, so the application doesn't crash.
    """
    global _nlp_model
    if _nlp_model is not None:
        return _nlp_model
    
    try:
        _nlp_model = spacy.load("en_core_web_sm")
        logger.info("✓ Loaded spaCy model: en_core_web_sm")
    except OSError:
        logger.warning("⚠ en_core_web_sm not found. Please run 'python -m spacy download en_core_web_sm'. Using blank model as fallback.")
        _nlp_model = spacy.blank("en")
    
    return _nlp_model


def anonymize_text(
    text: str,
    strategy: Literal['redact', 'synthetic'] = 'redact',
    include_pii_types: Optional[List[str]] = None,
    audit: bool = True
) -> Tuple[str, Dict]:
    """
    Anonymize PII in text using a two-pass approach: spaCy NER + regex patterns.
    
    This is the main function for anonymization. It takes a string of text
    and returns the anonymized version along with a detailed log of what was changed.
    
    Args:
        text: The original input text from the user.
        strategy: Defines how to mask data. 'redact' is the default.
        include_pii_types: Allows specifying exactly which PII types to look for.
        audit: If True, generates a detailed log of the anonymization process.
    
    Returns:
        A tuple containing:
        - The anonymized text string.
        - An audit_log dictionary with details of the operation.
    """
    if not text or not isinstance(text, str):
        return text, {"total_masked": 0, "by_type": {}, "masked_entities": []}
    
    # If no specific PII types are requested, we default to all configured types.
    if include_pii_types is None:
        all_pii_types = list(PII_ENTITY_CONFIG.keys()) + list(ADDITIONAL_PII_PATTERNS.keys())
        include_pii_types = all_pii_types
    
    nlp = _load_spacy_model()
    audit_log = {"total_masked": 0, "by_type": {}, "masked_entities": [], "timestamp": datetime.utcnow().isoformat()}
    
    anonymized_text = text
    
    # --- STEP 1: spaCy NER-based anonymization ---
    # First, we use the powerful spaCy model to find general entities like
    # names (PERSON), locations (GPE), organizations (ORG), and money (MONEY).
    try:
        doc = nlp(text)
        replacements = []
        
        for ent in doc.ents:
            ent_type = ent.label_
            
            if ent_type in include_pii_types and ent_type in PII_ENTITY_CONFIG:
                replacement_token = PII_ENTITY_CONFIG[ent_type]["token"]
                replacements.append((ent.start_char, ent.end_char, replacement_token, ent_type, ent.text))
        
        # We apply the replacements in reverse order of their appearance in the text.
        # This is a clever trick to avoid messing up the character positions of
        # entities that appear later in the sentence.
        for start, end, token, ent_type, original in reversed(replacements):
            anonymized_text = anonymized_text[:start] + token + anonymized_text[end:]
            
            # We log every single change for auditing purposes.
            audit_log["total_masked"] += 1
            audit_log["by_type"][ent_type] = audit_log["by_type"].get(ent_type, 0) + 1
            audit_log["masked_entities"].append({
                "type": ent_type, "original": original, "replacement": token
            })
        
        logger.debug(f"spaCy NER masked {len(replacements)} entities")
    
    except Exception as e:
        logger.error(f"Error during spaCy NER-based anonymization: {e}")
    
    # --- STEP 2: Regex-based anonymization for missed patterns ---
    # After the ML model has done its work, we do a second pass with regex
    # to catch specific patterns like emails and phone numbers.
    for pii_type, (pattern, replacement) in ADDITIONAL_PII_PATTERNS.items():
        if pii_type in include_pii_types:
            matches = list(re.finditer(pattern, anonymized_text))
            for match in reversed(matches):
                original_text = match.group()
                # We check to make sure we are not redacting something that's already been redacted.
                if "[REDACTED_" not in original_text:
                    anonymized_text = anonymized_text[:match.start()] + replacement + anonymized_text[match.end():]
                    audit_log["total_masked"] += 1
                    audit_log["by_type"][pii_type] = audit_log["by_type"].get(pii_type, 0) + 1
                    audit_log["masked_entities"].append({
                        "type": pii_type, "original": original_text, "replacement": replacement
                    })
    
    logger.info(f"Anonymization complete: {audit_log['total_masked']} entities masked in total.")
    
    return anonymized_text, audit_log


def batch_anonymize(
    texts: List[str],
    strategy: str = 'redact',
    audit: bool = True
) -> List[Tuple[str, Dict]]:
    """
    Anonymize a list of texts efficiently.
    This is useful for processing large amounts of data at once.
    """
    return [anonymize_text(text, strategy, audit=audit) for text in texts]


# This block runs if you execute the script directly (e.g., `python app/anonymizer.py`).
# It's a quick way to test that the basic functionality is working.
if __name__ == "__main__":
    _load_spacy_model() # Ensure model is downloaded and loaded
    test_text = "My name is John Doe, my email is test@example.com, and I want to pay my $500 premium."
    result, log = anonymize_text(test_text)
    print(f"Original: {test_text}")
    print(f"Anonymized: {result}")
    print(f"Audit Log: {log}")
    assert "[REDACTED_PERSON]" in result, "Name not masked!"
    assert "[REDACTED_EMAIL]" in result, "Email not masked!"
    assert "[REDACTED_AMOUNT]" in result, "Amount not masked!"
    print("\n✓ All basic anonymization tests passed!")
