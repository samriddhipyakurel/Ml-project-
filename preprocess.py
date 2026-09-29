import re
import string

def clean_text(text: str) -> str:
    """
    Cleans and preprocesses a raw string of text.
    1. Converts text to lowercase.
    2. Removes HTML tags.
    3. Removes URLs.
    4. Removes punctuation.
    5. Removes digits.
    6. Strips leading, trailing, and extra whitespaces.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text