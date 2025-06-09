"""Information Retrieval System Package"""

# Package version
__version__ = '1.0.0'

# Import main components for easy access
from .models import vector_space, lsa_model
from .preprocessing import preprocessing
from .indexing import inverted_index
from .evaluation import evaluator

# Define what should be imported with "from package import *"
__all__ = [
    'vector_space',
    'lsa_model', 
    'preprocessing',
    'inverted_index',
    'evaluator'
]
