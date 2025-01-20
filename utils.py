import re

class Cleaner:
    """
    A utility class for cleaning and preprocessing text and sentences.
    
    This class provides methods to clean text by removing unwanted characters, 
    such as HTML tags, URLs, special characters, and extra spaces, as well as 
    for converting sentences to uppercase and removing special characters.
    """
    
    @staticmethod
    def clean_text(text):
        """
        Cleans a given text by removing unwanted characters and formatting it.

        This method removes:
        - HTML tags
        - URLs
        - Special characters
        - Extra spaces
        
        Additionally, it trims leading and trailing whitespace.

        Args:
            text (str): The input text to be cleaned.

        Returns:
            str: The cleaned text.
        """
        # Step 1: Remove HTML tags (e.g., <p>, <div>)
        text = re.sub(r'<[^>]*?>', '', text)

        # Step 2: Remove URLs (e.g., https://example.com)
        text = re.sub(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            '', 
            text
        )

        # Step 3: Remove special characters (e.g., #, @, *, etc.)
        text = re.sub(r'[^a-zA-Z0-9 ]', '', text)

        # Step 4: Replace multiple consecutive spaces with a single space
        text = re.sub(r'\s{2,}', ' ', text)

        # Step 5: Trim leading and trailing whitespace
        text = text.strip()

        # Step 6: Remove any remaining extra spaces
        text = ' '.join(text.split())

        return text

    @staticmethod
    def clean_sentence(sentence):
        """
        Cleans and processes a given sentence by removing special characters 
        and converting it to uppercase.

        Args:
            sentence (str): The input sentence to be cleaned.

        Returns:
            str: The cleaned and uppercase sentence.
        """
        # Step 1: Remove all special characters using regex
        cleaned_sentence = re.sub(r'[^a-zA-Z0-9\s]', '', sentence)

        # Step 2: Convert the cleaned sentence to uppercase
        uppercase_sentence = cleaned_sentence.upper()

        return uppercase_sentence
