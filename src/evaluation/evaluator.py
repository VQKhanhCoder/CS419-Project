import numpy as np
from typing import List, Dict, Set, Tuple
import os
import sys
import logging
import json

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Now we can import from src
from src.models.vector_space import VectorSpaceModel
from src.models.lsa_model import LSAModel
from src.models.whoosh_model import WhooshModel

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self):
        """Initialize evaluator"""
        self.relevance_judgments: Dict[str, Set[int]] = {}
        self.logger = logging.getLogger(__name__)
    
    def load_relevance_judgments(self, res_dir: str):
        """Load relevance judgments from RES folder"""
        self.relevance_judgments.clear()
        try:
            for filename in os.listdir(res_dir):
                if filename.endswith('.txt'):
                    query_id = filename.split('.')[0]
                    relevant_docs = {}
                    
                    with open(os.path.join(res_dir, filename), 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                # Format: query_id doc_id relevance_score
                                parts = line.strip().split()
                                if len(parts) >= 3:
                                    doc_id = int(parts[1])
                                    relevance = int(parts[2])
                                    if relevance > 0:  # Chỉ lấy các documents có relevance > 0
                                        relevant_docs[doc_id] = relevance  # Lưu cả relevance score
                            except ValueError:
                                continue
                                
                    if relevant_docs:
                        self.relevance_judgments[query_id] = relevant_docs
            
            logger.info(f"Loaded relevance judgments for {len(self.relevance_judgments)} queries")
            
        except Exception as e:
            logger.error(f"Error loading relevance judgments: {str(e)}")
            raise

    def interpolate_precision(self, precision: List[float], recall: List[float]) -> Dict[float, float]:
        """Calculate interpolated precision at 11 standard recall levels (TREC)"""
        standard_recalls = np.linspace(0, 1, 11)  # 0, 0.1, ..., 1.0
        interpolated = {}
        
        for recall_level in standard_recalls:
            # Find all precision values at recall >= recall_level
            prec_at_recall = [p for p, r in zip(precision, recall) if r >= recall_level]
            if not prec_at_recall:  # No precision at this recall level
                interpolated[float(recall_level)] = 0.0
            else:
                # Maximum precision at recall >= recall_level
                interpolated[float(recall_level)] = max(prec_at_recall)
                
        return interpolated

    def calculate_map(self, results: List[Dict], relevant_docs: Dict[int, int]) -> float:
        """Calculate Mean Average Precision (MAP)"""
        if not results or not relevant_docs:
            return 0.0
            
        relevant_count = len(relevant_docs)
        precision_sum = 0.0
        relevant_found = 0
        
        for i, result in enumerate(results, 1):
            doc_id = result['doc_id']
            if doc_id in relevant_docs:
                relevant_found += 1
                precision_sum += relevant_found / i
                
        return precision_sum / relevant_count if relevant_count > 0 else 0.0

    def calculate_precision_recall(self, results: List[Dict], relevant_docs: Dict[int, int]) -> Tuple[List[float], List[float]]:
        """Calculate precision-recall curves"""
        precision = []
        recall = []
        
        relevant_found = 0
        total_relevant = len(relevant_docs)
        
        for i, result in enumerate(results, 1):
            if result['doc_id'] in relevant_docs:
                relevant_found += 1
                
            # Calculate precision and recall at each point
            precision.append(relevant_found / i)
            recall.append(relevant_found / total_relevant if total_relevant > 0 else 0.0)
            
        return precision, recall

    def evaluate_results(self, query_id: str, results: List[Dict], documents: Dict[int, str]) -> Dict:
        """Evaluate search results according to TREC metrics"""
        if not query_id in self.relevance_judgments:
            logger.warning(f"No relevance judgments found for query {query_id}")
            return {}
            
        if not results:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'map': 0.0,
                'interpolated_precision': {float(r): 0.0 for r in np.linspace(0, 1, 11)},
                'relevant_count': len(self.relevance_judgments[query_id]),
                'retrieved_count': 0,
                'relevant_retrieved': 0
            }

        try:
            # Không cần normalize results format nếu mô hình đã trả về đúng format
            relevant_docs = self.relevance_judgments[query_id]
            
            # Calculate basic counts
            retrieved_count = len(results)
            relevant_count = len(relevant_docs)
            relevant_retrieved = sum(1 for r in results if r['doc_id'] in relevant_docs)

            # Calculate precision-recall curves
            precision, recall = self.calculate_precision_recall(results, relevant_docs)
            
            # Calculate MAP
            map_score = self.calculate_map(results, relevant_docs)
            
            # Calculate interpolated precision
            interpolated = self.interpolate_precision(precision, recall)
            
            return {
                'precision': float(precision[-1]) if precision else 0.0,
                'recall': float(recall[-1]) if recall else 0.0,
                'map': float(map_score),
                'interpolated_precision': {float(k): float(v) for k, v in interpolated.items()},
                'relevant_count': relevant_count,
                'retrieved_count': retrieved_count,
                'relevant_retrieved': relevant_retrieved,
                'precision_curve': [float(p) for p in precision],
                'recall_curve': [float(r) for r in recall]
            }
            
        except Exception as e:
            logger.error(f"Error in evaluate_results: {str(e)}")
            return {}

    def evaluate_model(self, model_name: str, model, queries: Dict[str, str], documents: Dict[int, str]) -> Dict:
        """Evaluate a model across all queries"""
        results = {
            'model': model_name,
            'queries': {},
            'average_metrics': {
                'map': 0.0,
                'interpolated_precision': {float(r): 0.0 for r in np.linspace(0, 1, 11)},
                'precision': 0.0,
                'recall': 0.0
            }
        }
        
        query_count = 0
        
        for query_id, query_text in queries.items():
            # Get model predictions
            model_results = model.query(query_text)
            
            # Evaluate results for this query
            query_metrics = self.evaluate_results(query_id, model_results, documents)
            if query_metrics:
                results['queries'][query_id] = query_metrics
                
                # Update average metrics
                results['average_metrics']['map'] += query_metrics['map']
                results['average_metrics']['precision'] += query_metrics['precision']
                results['average_metrics']['recall'] += query_metrics['recall']
                
                # Update interpolated precision
                for recall_level, precision in query_metrics['interpolated_precision'].items():
                    results['average_metrics']['interpolated_precision'][recall_level] += precision
                    
                query_count += 1
        
        # Calculate final averages
        if query_count > 0:
            results['average_metrics']['map'] /= query_count
            results['average_metrics']['precision'] /= query_count
            results['average_metrics']['recall'] /= query_count
            
            for recall_level in results['average_metrics']['interpolated_precision']:
                results['average_metrics']['interpolated_precision'][recall_level] /= query_count
        
        return results

def save_evaluation_results(results: Dict, output_path: str):
    """Save evaluation results to file"""
    try:
        with open(output_path, 'w') as f:
            # Write header
            f.write("=" * 50 + "\n")
            f.write("CRANFIELD EVALUATION RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            # Write results for each model
            for model_name, model_results in results.items():
                f.write(f"\n{model_name}:\n")
                f.write("-" * 30 + "\n")
                metrics = model_results['average_metrics']
                
                # Write metrics
                f.write(f"Mean Average Precision (MAP): {metrics['map']:.3f}\n")
                f.write(f"Average Precision: {metrics['precision']:.3f}\n")
                f.write(f"Average Recall: {metrics['recall']:.3f}\n\n")
                
                f.write("Interpolated Precision at Standard Recall Levels:\n")
                for recall, precision in sorted(metrics['interpolated_precision'].items()):
                    f.write(f"  Recall {recall:.1f}: {precision:.3f}\n")
                
                f.write("\nPer-Query Results:\n")
                for query_id, query_metrics in sorted(model_results['queries'].items()):
                    f.write(f"\nQuery {query_id}:\n")
                    f.write(f"  Precision: {query_metrics['precision']:.3f}\n")
                    f.write(f"  Recall: {query_metrics['recall']:.3f}\n")
                    f.write(f"  MAP: {query_metrics['map']:.3f}\n")
                    
                f.write("\n" + "=" * 50 + "\n")
            
            # Write comparison
            f.write("\nMODEL COMPARISON:\n")
            maps = {name: results[name]['average_metrics']['map'] 
                   for name in ["Vector Space Model", "LSA Model", "Whoosh Model"]}
            
            for name, map_score in maps.items():
                f.write(f"{name} MAP: {map_score:.3f}\n")
            
            # Compare differences
            f.write("\nDifferences:\n")
            f.write(f"LSA vs VSM: {maps['LSA Model'] - maps['Vector Space Model']:.3f}\n")
            f.write(f"Whoosh vs VSM: {maps['Whoosh Model'] - maps['Vector Space Model']:.3f}\n")
            f.write(f"Whoosh vs LSA: {maps['Whoosh Model'] - maps['LSA Model']:.3f}\n")
            
        logger.info(f"Evaluation results saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving evaluation results: {str(e)}")

def main():
    """Evaluate VSM, LSA and Whoosh models on Cranfield dataset"""
    
    # Setup paths
    data_dir = os.path.join(project_root, 'data')
    index_dir = os.path.join(data_dir, 'Indexing')
    evaluation_dir = os.path.join(project_root, 'src', 'evaluation', 'results')
    os.makedirs(evaluation_dir, exist_ok=True)
    
    # Load inverted index
    index_path = os.path.join(index_dir, 'inverted_index.json')
    try:
        with open(index_path, 'r') as f:
            index_data = json.load(f)
            logger.info(f"Loaded inverted index with {len(index_data['vocabulary'])} terms")
    except Exception as e:
        logger.error(f"Error loading index: {str(e)}")
        return

    # Load documents (using doc IDs from index)
    documents = {}
    cranfield_dir = os.path.join(data_dir, 'Cranfield')
    doc_ids = sorted([int(id) for id in index_data["doc_lengths"].keys()])
    
    for doc_id in doc_ids:
        filename = f"{doc_id}.txt"
        try:
            with open(os.path.join(cranfield_dir, filename), 'r', encoding='utf-8') as f:
                documents[doc_id] = f.read().strip()
        except Exception as e:
            logger.error(f"Error loading document {filename}: {str(e)}")
    
    # Load queries
    queries = {}
    query_file = os.path.join(data_dir, 'TEST', 'query.txt')
    with open(query_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                query_id, text = parts
                queries[query_id] = text

    # Initialize all three models
    vsm_model = VectorSpaceModel()
    lsa_model = LSAModel()
    whoosh_model = WhooshModel()
    
    # Pass index data to models
    vsm_model.index_data = index_data
    lsa_model.index_data = index_data
    
    # Build vectors/matrix from index
    vsm_model.build_document_vectors()
    lsa_model.build_term_doc_matrix()
    
    # Build Whoosh index if needed
    whoosh_model.build_index(documents)
    
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.load_relevance_judgments(os.path.join(data_dir, 'TEST', 'RES'))
    
    # Evaluate all models
    print("\nEvaluating Vector Space Model...")
    vsm_results = evaluator.evaluate_model("VSM", vsm_model, queries, documents)
    
    print("\nEvaluating LSA Model...")
    lsa_results = evaluator.evaluate_model("LSA", lsa_model, queries, documents)
    
    print("\nEvaluating Whoosh Model...")
    whoosh_results = evaluator.evaluate_model("Whoosh", whoosh_model, queries, documents)
    
    # Combine results
    all_results = {
        "Vector Space Model": vsm_results,
        "LSA Model": lsa_results,
        "Whoosh Model": whoosh_results
    }
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(evaluation_dir, f'evaluation_results_{timestamp}.txt')
    save_evaluation_results(all_results, output_path)
    
    # Print summary to console
    print("\nEvaluation completed!")
    print(f"Detailed results saved to: {output_path}")
    
    # Print brief comparison
    print("\nQuick Comparison (MAP Scores):")
    print("-" * 30)
    models = ["Vector Space Model", "LSA Model", "Whoosh Model"]
    for model in models:
        map_score = all_results[model]['average_metrics']['map']
        print(f"{model}: {map_score:.3f}")

if __name__ == "__main__":
    from datetime import datetime
    main()
