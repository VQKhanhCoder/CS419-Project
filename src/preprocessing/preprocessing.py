import spacy
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
import re
import logging
from nltk.corpus import wordnet
import json
from pathlib import Path
from typing import List

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Initialize lemmatizer and load spaCy model
lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_lg")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_phrases_spacy(text):
    """
    Use spaCy to detect noun phrases and replace them with single tokens.
    :param text: String, input text.
    :return: List of detected phrases.
    """
    doc = nlp(text)
    phrases = []
    for chunk in doc.noun_chunks:
        phrases.append(chunk.text.lower())
    return phrases

def tokenize(text):
    """
    Tokenize the input text into words using spaCy.
    :param text: String, input text.
    :return: List of tokens.
    """
    doc = nlp(text)
    return [token.text.lower() for token in doc if not token.is_punct]

def normalize_text(text):
    """Cải thiện normalize để giữ lại nhiều thông tin hơn"""
    # Giữ lại các ký tự đặc biệt quan trọng và số
    text = re.sub(r'[^a-zA-Z0-9\s\-_./()]', ' ', text)
    # Xử lý dấu chấm và gạch ngang đặc biệt
    text = re.sub(r'\.(?!\d)', ' ', text)  # Giữ lại dấu chấm trong số
    text = re.sub(r'-', ' ', text)  # Tách các từ ghép bằng gạch ngang
    # Thêm xử lý đặc biệt cho ký hiệu toán học và số
    text = re.sub(r'(\d+)([a-zA-Z])', r'\1 \2', text)  # Tách số và chữ
    text = re.sub(r'([a-zA-Z])(\d+)', r'\1 \2', text)  # Tách chữ và số
    # Chuẩn hóa khoảng trắng
    text = ' '.join(text.split())
    return text.lower().strip()

def remove_stopwords(tokens):
    """
    Remove stopwords but preserve technical terms
    """
    stop_words = set(stopwords.words('english'))
    # Remove common technical words from stopwords
    technical_terms = {'no', 'not', 'between', 'under', 'above', 'below'}
    stop_words = stop_words - technical_terms
    return [token for token in tokens if token not in stop_words or len(token) > 2]

def get_wordnet_pos(word, tag):
    """Map POS tag to WordNet POS tag"""
    tag_map = {
        'JJ': wordnet.ADJ,
        'JJR': wordnet.ADJ,
        'JJS': wordnet.ADJ,
        'NN': wordnet.NOUN,
        'NNS': wordnet.NOUN,
        'NNP': wordnet.NOUN,
        'NNPS': wordnet.NOUN,
        'RB': wordnet.ADV,
        'RBR': wordnet.ADV,
        'RBS': wordnet.ADV,
        'VB': wordnet.VERB,
        'VBD': wordnet.VERB,
        'VBG': wordnet.VERB,
        'VBN': wordnet.VERB,
        'VBP': wordnet.VERB,
        'VBZ': wordnet.VERB,
    }
    return tag_map.get(tag, wordnet.NOUN)  # Default to NOUN if tag not found

def lemmatize_token(token):
    """Lemmatize token using correct POS"""
    pos = get_wordnet_pos(token.text, token.tag_)
    return lemmatizer.lemmatize(token.text.lower(), pos=pos)

def preprocess_text(text, use_lemma=True):
    """Simplified preprocessing pipeline for both documents and queries"""
    try:
        if not text or not isinstance(text, str):
            return []

        # Normalize text
        text = normalize_text(text)
        doc = nlp(text)
        
        # Process tokens
        processed_tokens = []
        for token in doc:
            if (token.pos_ in {'NOUN', 'ADJ', 'VERB'} and 
                len(token.text) > 2 and 
                not token.is_stop):
                
                word = token.text.lower()
                if use_lemma:
                    word = lemmatize_token(token)
                
                if len(word) > 2:
                    processed_tokens.append(word)
        
        # Remove duplicates preserving order
        seen = set()
        final_tokens = [x for x in processed_tokens if not (x in seen or seen.add(x))]
        
        logger.info(f"Extracted {len(final_tokens)} tokens from text length {len(text)}")
        return final_tokens
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        return []

def build_vocabulary(documents: List[str]) -> List[str]:
    """Build vocabulary from all documents"""
    vocabulary = set()
    for doc in documents:
        tokens = preprocess_text(doc)
        if tokens:
            vocabulary.update(tokens)
    return sorted(list(vocabulary))

def save_vocabulary(vocabulary: List[str], filepath: str):
    """Save vocabulary to JSON file"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({"vocabulary": vocabulary}, f, indent=2)
        logger.info(f"Saved vocabulary with {len(vocabulary)} terms")
    except Exception as e:
        logger.error(f"Error saving vocabulary: {str(e)}")

def load_vocabulary(filepath: str) -> List[str]:
    """Load vocabulary from JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("vocabulary", [])
    except Exception as e:
        logger.error(f"Error loading vocabulary: {str(e)}")
        return []

def main():
    """Simple preprocessing demo"""
    print("\nCranfield Document Preprocessing Demo")
    print("-" * 50)
    
    while True:
        text = input("\nEnter text (or 'q' to quit): ")
        if text.lower() == 'q':
            break
            
        if not text:
            print("Please enter some text!")
            continue
        
        # Process text and show tokens
        tokens = preprocess_text(text)
        print("\nTokens:")
        for i, token in enumerate(tokens, 1):
            print(f"{i:2d}. {token}")
        
        print(f"\nTotal tokens: {len(tokens)}")

if __name__ == "__main__":
    main()