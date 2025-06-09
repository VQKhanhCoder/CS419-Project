import os
import logging
from typing import Dict
import pandas as pd
from preprocessing import build_vocabulary, save_vocabulary

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_cranfield_documents(data_dir: str) -> Dict[int, str]:
    """Load Cranfield documents from local folder"""
    documents = {}
    cranfield_dir = os.path.join(data_dir, "Cranfield")
    
    try:
        for filename in os.listdir(cranfield_dir):
            if filename.endswith('.txt'):
                doc_id = int(filename.split('.')[0])
                with open(os.path.join(cranfield_dir, filename), 'r', encoding='utf-8') as f:
                    documents[doc_id] = f.read().strip()
        logger.info(f"Loaded {len(documents)} documents from Cranfield folder")
        return documents
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        return {}

def build_cranfield_vocabulary(data_dir: str):
    """Build and save vocabulary from Cranfield documents"""
    try:
        # Load documents
        documents = load_cranfield_documents(data_dir)
        if not documents:
            raise ValueError("No documents loaded")
            
        # Convert documents to list
        doc_texts = list(documents.values())
        logger.info(f"Building vocabulary from {len(doc_texts)} documents")
        
        # Build vocabulary
        vocabulary = build_vocabulary(doc_texts)
        logger.info(f"Built vocabulary with {len(vocabulary)} terms")
        
        # Save vocabulary
        vocab_dir = os.path.dirname(__file__)
        vocab_path = os.path.join(vocab_dir, 'vocabulary.json')
        save_vocabulary(vocabulary, vocab_path)
        
        # Create and save vocabulary statistics
        vocab_df = pd.DataFrame({
            'term': vocabulary,
            'length': [len(term) for term in vocabulary]
        })
        stats_path = os.path.join(vocab_dir, 'vocabulary_stats.csv')
        vocab_df.to_csv(stats_path, index=False)
        logger.info(f"Saved vocabulary statistics to {stats_path}")
        
        return vocabulary
        
    except Exception as e:
        logger.error(f"Error building vocabulary: {str(e)}")
        return []

def main():
    """Main function to build vocabulary"""
    # Get project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_dir = os.path.join(project_root, 'data')
    
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        return
        
    vocabulary = build_cranfield_vocabulary(data_dir)
    if vocabulary:
        print(f"Built vocabulary with {len(vocabulary)} terms")
        print("Sample terms:", vocabulary[:10])
    else:
        print("Failed to build vocabulary")

if __name__ == "__main__":
    main()
