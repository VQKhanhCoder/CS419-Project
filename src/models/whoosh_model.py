import os
from typing import List, Dict
import logging
from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
from whoosh.index import create_in, open_dir
from whoosh.analysis import StandardAnalyzer

logger = logging.getLogger(__name__)

class WhooshModel:
    def __init__(self):
        self.index_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'WhooshIndex')
        self.index = None
        self.analyzer = StandardAnalyzer()
        self.schema = Schema(
            doc_id=ID(stored=True),
            content=TEXT(analyzer=self.analyzer)
        )
        
    def build_index(self, documents: Dict[int, str]):
        """Build Whoosh index from documents"""
        try:
            # Create index directory if not exists
            os.makedirs(self.index_dir, exist_ok=True)
            
            # Create new index
            self.index = create_in(self.index_dir, self.schema)
            writer = self.index.writer()
            
            # Add documents
            for doc_id, content in documents.items():
                writer.add_document(
                    doc_id=str(doc_id),
                    content=content
                )
            
            writer.commit()
            logger.info(f"Built Whoosh index with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error building Whoosh index: {str(e)}")
            raise

    def load_index(self):
        """Load existing Whoosh index"""
        try:
            if os.path.exists(self.index_dir):
                self.index = open_dir(self.index_dir)
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading Whoosh index: {str(e)}")
            return False

    def query(self, query_text: str, top_k: int = 10) -> List[Dict]:
        """Search documents using Whoosh"""
        try:
            if not self.index:
                if not self.load_index():
                    raise ValueError("No index available")
            
            # Create parser and search
            parser = QueryParser("content", self.index.schema)
            searcher = self.index.searcher()
            
            # Parse query and search
            query = parser.parse(query_text)
            results = searcher.search(query, limit=top_k)
            
            # Format results to match evaluator requirements
            formatted_results = []
            for hit in results:
                formatted_results.append({
                    'doc_id': int(hit['doc_id']),
                    'score': float(hit.score),
                    'source': 'Whoosh'
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in query: {str(e)}")
            return []

def main():
    """Test Whoosh model with sample queries"""
    try:
        # Load sample documents
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        cranfield_dir = os.path.join(data_dir, 'Cranfield')
        documents = {}
        
        print("\nLoading documents...")
        for filename in sorted(os.listdir(cranfield_dir)):
            if filename.endswith('.txt'):
                doc_id = int(filename.split('.')[0])
                with open(os.path.join(cranfield_dir, filename), 'r', encoding='utf-8') as f:
                    documents[doc_id] = f.read().strip()
        
        # Initialize and build index
        print(f"\nBuilding Whoosh index for {len(documents)} documents...")
        model = WhooshModel()
        model.build_index(documents)
        
        # Test queries
        test_queries = [
            "flow of compressible fluid",
            "shock wave interaction",
            "boundary layer equations",
            "heat transfer in supersonic flow",
            "aerodynamic noise AND turbulence"
        ]
        
        print("\nTesting sample queries:")
        print("-" * 50)
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            results = model.query(query, top_k=5)
            
            if results:
                print("\nTop 5 results:")
                for rank, hit in enumerate(results, 1):
                    doc_id = hit['doc_id']
                    score = hit['score']
                    print(f"#{rank} Doc {doc_id}: {score:.4f}")
                    # Show snippet of document content
                    doc_snippet = documents[doc_id][:200] + "..."
                    print(f"Snippet: {doc_snippet}\n")
            else:
                print("No results found")
                
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
