import re

def anonymize_text(text: str) -> str:
    """
    Anonymizes PII in a given text.

    Args:
        text: The text to anonymize.

    Returns:
        The anonymized text.
    """
    # This is a simple anonymizer that replaces names and amounts using regex.
    # A more robust solution would use a proper NER model to identify PII.

    # Regex for a simple name structure (e.g., "John Doe")
    anonymized_text = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[REDACTED_PERSON]', text)
    
    # Regex to find currency symbols followed by numbers
    anonymized_text = re.sub(r'[\$€£]\d+(\.\d{2})?', '[REDACTED_AMOUNT]', anonymized_text)
        
    return anonymized_text
