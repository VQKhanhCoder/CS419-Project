import os
import sys
import json
import logging
from typing import Dict, List
from collections import defaultdict

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.preprocessing.preprocessing import preprocess_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InvertedIndex:
    """Simple inverted index with position information"""
    def __init__(self):
        self.index = defaultdict(list)  # term -> [(doc_id, freq, [positions])]
        self.doc_lengths = {}  # doc_id -> length
        self.vocabulary = set()  # unique terms

    def add_document(self, doc_id: int, text: str) -> None:
        """Add a document to the index"""
        # Preprocess document
        tokens = preprocess_text(text)
        
        # Update document length
        self.doc_lengths[doc_id] = len(tokens)
        
        # Add terms to index with positions
        positions = defaultdict(list)
        for pos, term in enumerate(tokens):
            positions[term].append(pos)
            self.vocabulary.add(term)
        
        # Add to inverted index with frequency and positions
        for term, pos_list in positions.items():
            self.index[term].append((doc_id, len(pos_list), pos_list))

    def save_index(self, filepath: str) -> None:
        """Save inverted index to JSON file"""
        try:
            index_data = {
                "index": {k: v for k, v in self.index.items()},
                "doc_lengths": self.doc_lengths,
                "vocabulary": list(self.vocabulary)
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2)
            logger.info(f"Saved index with {len(self.vocabulary)} terms")
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")

    def load_index(self, filepath: str) -> None:
        """Load inverted index from JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.index = defaultdict(list, data["index"])
                self.doc_lengths = data["doc_lengths"]
                self.vocabulary = set(data["vocabulary"])
            logger.info(f"Loaded index with {len(self.vocabulary)} terms")
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")

def create_index_from_files(data_dir: str = None) -> None:
    """Create and save index from Cranfield documents"""
    try:
        # Get project root if data_dir not provided
        if not data_dir:
            data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
        
        cranfield_dir = os.path.join(data_dir, 'Cranfield')
        indexing_dir = os.path.join(data_dir, 'Indexing')
        
        # Create Indexing directory if it doesn't exist
        os.makedirs(indexing_dir, exist_ok=True)
        
        # Initialize index
        index = InvertedIndex()
        documents_processed = 0
        
        # Process each document
        for filename in sorted(os.listdir(cranfield_dir)):
            if filename.endswith('.txt'):
                try:
                    doc_id = int(filename.split('.')[0])
                    with open(os.path.join(cranfield_dir, filename), 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        index.add_document(doc_id, text)
                        documents_processed += 1
                        if documents_processed % 100 == 0:
                            logger.info(f"Processed {documents_processed} documents...")
                except Exception as e:
                    logger.error(f"Error processing document {filename}: {str(e)}")
                    continue
        
        # Save index
        index_path = os.path.join(indexing_dir, 'inverted_index.json')
        index.save_index(index_path)
        
        logger.info(f"Successfully indexed {documents_processed} documents")
        logger.info(f"Index saved to {index_path}")
        logger.info(f"Vocabulary size: {len(index.vocabulary)}")
        
    except Exception as e:
        logger.error(f"Error creating index: {str(e)}")

if __name__ == "__main__":
    create_index_from_files()
