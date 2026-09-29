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

if __name__ == "__main__":
    # Test suite for clean_text
    test_cases = [
        ("Hello World!", "hello world"),
        ("Check this: https://example.com 123", "check this"),
        ("<b>HTML Tags</b> test", "html tags test"),
        ("   Extra    spaces   ", "extra spaces")
    ]
    
    for raw, expected in test_cases:
        result = clean_text(raw)
        assert result == expected, f"Failed for '{raw}': got '{result}', expected '{expected}'"
    print("All preprocessing tests passed successfully!")