import os
import sys
import json
import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Tuple, Dict
import logging
from collections import Counter
from sklearn.preprocessing import normalize

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.preprocessing.preprocessing import preprocess_text


logger = logging.getLogger(__name__)

class VectorSpaceModel:
    def __init__(self):
        self.vocabulary = []
        self.vocab_index = {}
        self.doc_vectors = None
        self.idf = None
        self.doc_ids = []
        self.index_data = None
        # Load resources ngay khi khởi tạo
        self.load_resources()
        # Build vectors từ inverted index
        self.build_document_vectors()

    def load_resources(self):
        """Load vocabulary và inverted index"""
        try:
            # Load vocabulary
            vocab_path = os.path.join(project_root, 'src', 'preprocessing', 'vocabulary.json')
            with open(vocab_path, 'r') as f:
                self.vocabulary = json.load(f)["vocabulary"]
                self.vocab_index = {word: idx for idx, word in enumerate(self.vocabulary)}
            
            # Load inverted index
            index_path = os.path.join(project_root, 'data', 'Indexing', 'inverted_index.json')
            with open(index_path, 'r') as f:
                self.index_data = json.load(f)
                
            logger.info(f"Loaded vocabulary with {len(self.vocabulary)} terms")
            return True
            
        except Exception as e:
            logger.error(f"Error loading resources: {str(e)}")
            return False

    def build_document_vectors(self):
        """Build document vectors từ inverted index"""
        try:
            num_docs = len(self.index_data["doc_lengths"])
            num_terms = len(self.vocabulary)
            
            # Calculate IDF from inverted index
            doc_freq = np.zeros(num_terms)
            for term, postings in self.index_data["index"].items():
                if term in self.vocab_index:
                    term_idx = self.vocab_index[term]
                    doc_freq[term_idx] = len(postings)  # Số document chứa term
            
            self.idf = np.log10((num_docs + 1) / (doc_freq + 1))
            
            # Build document vectors from postings lists
            self.doc_ids = sorted(list(map(int, self.index_data["doc_lengths"].keys())))
            rows, cols, data = [], [], []
            
            for term, postings in self.index_data["index"].items():
                if term in self.vocab_index:
                    term_idx = self.vocab_index[term]
                    for doc_id, freq, _ in postings:
                        doc_idx = self.doc_ids.index(doc_id)
                        doc_length = self.index_data["doc_lengths"][str(doc_id)]
                        tfidf = (freq/doc_length) * self.idf[term_idx]
                        rows.append(doc_idx)
                        cols.append(term_idx)
                        data.append(tfidf)

            # Create sparse document-term matrix
            self.doc_vectors = csr_matrix((data, (rows, cols)), 
                                        shape=(len(self.doc_ids), num_terms))
            
            # Normalize document vectors 
            self.doc_vectors = normalize(self.doc_vectors, norm='l2', axis=1)
            
            logger.info(f"Built document vectors matrix of shape {self.doc_vectors.shape}")
            return True
                
        except Exception as e:
            logger.error(f"Error building document vectors: {str(e)}")
            return False

    def vectorize_query(self, query: str) -> np.ndarray:
        """Convert query to TF-IDF vector"""
        try:
            # Get query term frequencies
            tokens = preprocess_text(query)
            if not tokens:
                return np.zeros(len(self.vocabulary))

            tf = Counter(tokens)
            query_vector = np.zeros(len(self.vocabulary))

            # Create TF-IDF vector
            query_length = len(tokens)
            for term, freq in tf.items():
                if term in self.vocab_index:
                    term_idx = self.vocab_index[term]
                    # Normalized TF * IDF
                    query_vector[term_idx] = (freq / query_length) * self.idf[term_idx]

            # Normalize to unit length
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm

            return query_vector

        except Exception as e:
            logger.error(f"Error vectorizing query: {str(e)}")
            return np.zeros(len(self.vocabulary))

    def clean_query(self, query_text: str) -> str:
        """Remove leading numbers from query"""
        return ' '.join(query_text.split()[1:]) if query_text.split() and query_text.split()[0].isdigit() else query_text

    def query(self, query_text: str, top_k: int = 10) -> List[Dict]:
        """Return top-k documents most similar to query"""
        try:
            # Clean query first then process
            cleaned_query = self.clean_query(query_text)
            query_vector = self.vectorize_query(cleaned_query)

            # Calculate cosine similarities with sparse matrix
            query_vector = query_vector.reshape(-1, 1)
            similarities = self.doc_vectors.dot(query_vector).flatten()
            
            # Get top k results
            top_k_indices = np.argsort(-similarities)[:top_k]

            # Format results to match evaluator requirements 
            results = []
            for idx in top_k_indices:
                doc_id = self.doc_ids[idx]
                score = float(similarities[idx])
                results.append({
                    'doc_id': doc_id,
                    'score': score,
                    'source': 'VSM'
                })

            return results

        except Exception as e:
            logger.error(f"Error in query: {str(e)}")
            return []

def main():
    """Test vector space model"""
    vsm = VectorSpaceModel()
    
    # Test query
    query = "papers on shock-sound wave interaction ."
    results = vsm.query(query, top_k=10)
    
    print(f"\nTop 10 documents for query: {query}")
    for rank, result in enumerate(results, 1):
        print(f"#{rank} Doc {result['doc_id']}: {result['score']:.4f}")

if __name__ == "__main__":
    main()
