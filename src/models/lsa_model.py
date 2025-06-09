import os
import sys
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize 
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import List, Dict
from collections import Counter
import re
from typing import Set

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Change relative imports to absolute imports
from src.preprocessing.preprocessing import preprocess_text, load_vocabulary
import json

logger = logging.getLogger(__name__)

class LSAModel:
    def __init__(self, n_components=300):  # Set default k=200
        self.k = n_components  # Số chiều cố định
        self.vocabulary = []
        self.vocab_index = {}
        self.term_doc_matrix = None
        self.doc_ids = []
        self.index_data = None
        
        # LSA components
        self.U_k = None      # Term-topic matrix
        self.Sigma_k = None  # Singular values 
        self.Vt_k = None     # Document-topic matrix
        
        # Load resources
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.load_resources()
        self.build_term_doc_matrix()

    def load_resources(self):
        """Load vocabulary và inverted index từ project root"""
        try:
            # Load vocabulary 
            vocab_path = os.path.join(self.project_root, 'src', 'preprocessing', 'vocabulary.json')
            with open(vocab_path, 'r') as f:
                vocab_data = json.load(f)
                self.vocabulary = vocab_data["vocabulary"]
                self.vocab_index = {word: idx for idx, word in enumerate(self.vocabulary)}
            
            # Load inverted index
            index_path = os.path.join(self.project_root, 'data', 'Indexing', 'inverted_index.json')
            with open(index_path, 'r') as f:
                self.index_data = json.load(f)
                
            logger.info(f"Loaded vocabulary with {len(self.vocabulary)} terms")
            
            # Build document vectors ngay sau khi load resources
            self.build_term_doc_matrix()
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading resources: {str(e)}")
            return False

    def build_term_doc_matrix(self) -> bool:
        """Build term-document matrix và thực hiện SVD với k=300"""
        try:
            num_docs = len(self.index_data["doc_lengths"])
            num_terms = len(self.vocabulary)
            self.doc_ids = sorted(list(map(int, self.index_data["doc_lengths"].keys())))

            # Build sparse term-document matrix
            rows, cols, data = [], [], []
            for term, postings in self.index_data["index"].items():
                if term in self.vocab_index:
                    term_idx = self.vocab_index[term]
                    for doc_id, freq, _ in postings:
                        doc_idx = self.doc_ids.index(doc_id)
                        doc_length = float(self.index_data["doc_lengths"][str(doc_id)])
                        tf = freq/doc_length if doc_length else 0
                        idf = np.log10(num_docs / (len(postings) + 1))
                        rows.append(term_idx)
                        cols.append(doc_idx)
                        data.append(tf * idf)

            # Create sparse matrix 
            self.term_doc_matrix = csr_matrix((data, (rows, cols)), 
                                           shape=(num_terms, num_docs))

            # Apply SVD with fixed k=300
            k = min(self.k, num_docs-1)
            svd = TruncatedSVD(n_components=k)
            self.U_k = svd.fit_transform(self.term_doc_matrix)
            self.Sigma_k = svd.singular_values_
            self.Vt_k = svd.components_

            # Calculate document vectors in LSA space
            self.docs_latent = self.Vt_k.T  # (num_docs, k)
            self.docs_latent = normalize(self.docs_latent, norm='l2', axis=1)

            # Calculate explained variance
            explained_var = svd.explained_variance_ratio_.sum()
            logger.info(f"Built LSA model with k={k} topics")
            logger.info(f"Explained variance ratio: {explained_var:.2%}")

            return True

        except Exception as e:
            logger.error(f"Error in build_term_doc_matrix: {str(e)}")
            return False

    def fit(self, documents, doc_ids):
        """Fit LSA model using documents"""
        try:
            if not self.vocabulary:
                vocab_path = os.path.join(self.project_root, 'src', 'preprocessing', 'vocabulary.json')
                with open(vocab_path, 'r') as f:
                    self.vocabulary = json.load(f)["vocabulary"]
                self.vocab_index = {word: idx for idx, word in enumerate(self.vocabulary)}
            
            # Build term-document matrix 
            if not self.build_term_doc_matrix():
                raise ValueError("Failed to build term-document matrix")
                
            # SVD is already done in build_term_doc_matrix()
            self.document_ids = self.doc_ids
            
            # Calculate explained variance
            total_variance = (self.Sigma_k ** 2).sum()
            explained_variance = (self.Sigma_k[:self.k] ** 2).sum() / total_variance
            
            logger.info(f"LSA model fit with {self.k} components")
            logger.info(f"Explained variance ratio: {explained_variance:.2%}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error fitting LSA model: {str(e)}")
            return False

    def transform_query(self, query_text: str) -> np.ndarray:
        """Transform query to LSA space"""
        try:
            # Build query vector in term space (terms x 1)
            query_vec = np.zeros(len(self.vocabulary))
            tokens = preprocess_text(query_text)
            term_freqs = Counter(tokens)
            doc_length = float(len(tokens))
            
            for term, freq in term_freqs.items():
                if term in self.vocab_index:
                    idx = self.vocab_index[term]
                    tf = freq / doc_length
                    idf = np.log10(len(self.doc_ids) / 
                                 (len(self.index_data["index"].get(term, [])) + 1))
                    query_vec[idx] = tf * idf

            # Project query to LSA space: q' = q^T * U_k * Sigma_k^-1
            query_latent = query_vec @ self.U_k @ np.diag(1.0/self.Sigma_k)
            
            # Normalize
            norm = np.linalg.norm(query_latent)
            if norm > 0:
                query_latent = query_latent / norm
            
            return query_latent

        except Exception as e:
            logger.error(f"Error in transform_query: {str(e)}")
            return np.zeros(self.k)

    def clean_query(self, query_text: str) -> str:
        """Remove leading numbers from query"""
        return ' '.join(query_text.split()[1:]) if query_text.split() and query_text.split()[0].isdigit() else query_text

    def parse_boolean_query(self, query_text: str):
        """Parse boolean query and return structured representation"""
        query_text = query_text.strip()
        
        # Handle complex NOT cases first
        if ' AND NOT ' in query_text.upper():
            parts = re.split(r'\s+AND\s+NOT\s+', query_text, flags=re.IGNORECASE)
            if len(parts) == 2:
                return {'type': 'AND_NOT', 'positive': parts[0].strip(), 'negative': parts[1].strip()}
        elif ' OR NOT ' in query_text.upper():
            parts = re.split(r'\s+OR\s+NOT\s+', query_text, flags=re.IGNORECASE)
            if len(parts) == 2:
                return {'type': 'OR_NOT', 'positive': parts[0].strip(), 'negative': parts[1].strip()}
        elif query_text.upper().startswith('NOT '):
            return {'type': 'NOT', 'terms': [query_text[4:].strip()]}
        elif ' AND ' in query_text.upper():
            parts = re.split(r'\s+AND\s+', query_text, flags=re.IGNORECASE)
            return {'type': 'AND', 'terms': [part.strip() for part in parts]}
        elif ' OR ' in query_text.upper():
            parts = re.split(r'\s+OR\s+', query_text, flags=re.IGNORECASE)
            return {'type': 'OR', 'terms': [part.strip() for part in parts]}
        else:
            return {'type': 'SIMPLE', 'terms': [query_text]}

    def get_documents_for_terms(self, terms: List[str]) -> Dict[str, Set[int]]:
        """Get documents containing specific terms from inverted index"""
        doc_sets = {}
        for term in terms:
            # Preprocess term first
            processed_terms = preprocess_text(term)
            term_docs = set()
            
            for processed_term in processed_terms:
                if processed_term in self.index_data["index"]:
                    postings = self.index_data["index"][processed_term]
                    term_docs.update(posting[0] for posting in postings)
            
            doc_sets[term] = term_docs
        return doc_sets

    def apply_boolean_filter(self, parsed_query: Dict, candidate_docs: Set[int]) -> Set[int]:
        """Apply boolean logic to filter documents"""
        if parsed_query['type'] == 'SIMPLE':
            return candidate_docs
            
        all_docs = set(int(doc_id) for doc_id in self.index_data["doc_lengths"].keys())
        
        if parsed_query['type'] == 'AND':
            # Intersection of all term document sets
            doc_sets = self.get_documents_for_terms(parsed_query['terms'])
            result = set.intersection(*doc_sets.values()) if doc_sets else set()
            
        elif parsed_query['type'] == 'OR':
            # Union of all term document sets
            doc_sets = self.get_documents_for_terms(parsed_query['terms'])
            result = set.union(*doc_sets.values()) if doc_sets else set()
            
        elif parsed_query['type'] == 'NOT':
            # All documents except those containing the term
            excluded_doc_sets = self.get_documents_for_terms(parsed_query['terms'])
            excluded_docs = set.union(*excluded_doc_sets.values()) if excluded_doc_sets else set()
            result = all_docs - excluded_docs
            
        elif parsed_query['type'] == 'AND_NOT':
            # Documents containing positive terms but not negative terms
            positive_doc_sets = self.get_documents_for_terms([parsed_query['positive']])
            negative_doc_sets = self.get_documents_for_terms([parsed_query['negative']])
            
            positive_docs = set.union(*positive_doc_sets.values()) if positive_doc_sets else set()
            negative_docs = set.union(*negative_doc_sets.values()) if negative_doc_sets else set()
            
            result = positive_docs - negative_docs
            
        elif parsed_query['type'] == 'OR_NOT':
            # Documents containing positive terms OR not containing negative terms
            positive_doc_sets = self.get_documents_for_terms([parsed_query['positive']])
            negative_doc_sets = self.get_documents_for_terms([parsed_query['negative']])
            
            positive_docs = set.union(*positive_doc_sets.values()) if positive_doc_sets else set()
            negative_docs = set.union(*negative_doc_sets.values()) if negative_doc_sets else set()
            
            result = positive_docs | (all_docs - negative_docs)
            
        else:
            result = candidate_docs
            
        return result & candidate_docs  # Intersect with semantic candidates

    def query(self, query_text: str, top_k: int = 10) -> List[Dict]:
        """Enhanced query with boolean search support"""
        try:
            # Clean query first
            cleaned_query = self.clean_query(query_text)
            
            # Parse boolean query
            parsed_query = self.parse_boolean_query(cleaned_query)
            
            # For semantic similarity, use the full query without Boolean operators
            semantic_query = cleaned_query
            if parsed_query['type'] in ['AND_NOT', 'OR_NOT']:
                semantic_query = parsed_query['positive']  # Use only positive part
            elif parsed_query['type'] == 'NOT':
                # For pure NOT queries, return empty or use all docs
                semantic_query = ""
            elif parsed_query['type'] in ['AND', 'OR']:
                semantic_query = ' '.join(parsed_query['terms'])
            
            # Get semantic similarity results
            if semantic_query.strip():
                query_latent = self.transform_query(semantic_query)
                similarities = self.docs_latent @ query_latent
                
                # Get expanded candidates for Boolean filtering
                top_indices = np.argsort(-similarities)
                candidate_docs = set(self.doc_ids[idx] for idx in top_indices[:top_k*5])
            else:
                # For pure NOT queries, use all documents as candidates
                similarities = np.zeros(len(self.doc_ids))
                candidate_docs = set(self.doc_ids)
                top_indices = list(range(len(self.doc_ids)))
            
            # Apply boolean filtering
            filtered_docs = self.apply_boolean_filter(parsed_query, candidate_docs)
            
            # Re-rank filtered documents by similarity
            results = []
            for idx in top_indices:
                doc_id = self.doc_ids[idx]
                if doc_id in filtered_docs:
                    results.append({
                        'doc_id': doc_id,
                        'score': float(similarities[idx]),
                        'source': 'LSA+Boolean'
                    })
                    if len(results) >= top_k:
                        break
            
            # If no results from boolean filtering, fallback to semantic only for simple queries
            if not results and parsed_query['type'] == 'SIMPLE' and semantic_query.strip():
                for idx in top_indices[:top_k]:
                    doc_id = self.doc_ids[idx]
                    results.append({
                        'doc_id': doc_id,
                        'score': float(similarities[idx]),
                        'source': 'LSA'
                    })
            
            return results

        except Exception as e:
            logger.error(f"Error in LSA boolean query: {str(e)}")
            return []

def main():
    """Demo LSA model với k=200"""
    from glob import glob
    import os

    # Load sample documents
    docs = []
    doc_ids = []
    data_dir = "../../data/Cranfield"
    
    for file in glob(os.path.join(data_dir, "*.txt")):
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            docs.append(text)
            doc_ids.append(int(os.path.basename(file).split('.')[0]))

    # Create and fit model
    model = LSAModel()  # Will use k=200 by default
    model.fit(docs, doc_ids)

    # Test query
    query = "flow AND pressure"
    results = model.query(query, top_k=5)
    
    print(f"\nTop 5 results for query: {query}")
    for r in results:
        print(f"Doc {r['doc_id']}: {r['score']:.4f}")

if __name__ == "__main__":
    main()
