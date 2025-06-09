import streamlit as st
import pandas as pd
import os
import sys
import logging
import json
from datetime import datetime

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.vector_space import VectorSpaceModel
from src.models.lsa_model import LSAModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def load_cranfield_documents():
    """Load Cranfield documents from local folder"""
    documents = {}
    data_dir = os.path.join(project_root, "data")
    cranfield_dir = os.path.join(data_dir, "Cranfield")
    
    try:
        for filename in sorted(os.listdir(cranfield_dir)):
            if filename.endswith('.txt'):
                doc_id = int(filename.split('.')[0])
                with open(os.path.join(cranfield_dir, filename), 'r', encoding='utf-8') as f:
                    documents[doc_id] = f.read().strip()
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return {}

@st.cache_resource
def load_cranfield_queries():
    """Load queries from local file"""
    queries = {}
    data_dir = os.path.join(project_root, "data")
    query_file = os.path.join(data_dir, "TEST", "query.txt")
    
    try:
        with open(query_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    query_id, text = parts
                    queries[query_id] = text
        logger.info(f"Loaded {len(queries)} queries")
        return queries
    except Exception as e:
        st.error(f"Error loading queries: {str(e)}")
        return {}

@st.cache_resource
def initialize_models(_documents):
    """Initialize VSM and LSA models"""
    try:
        st.info("Initializing models... This may take a moment.")
        
        # Initialize models
        vsm_model = VectorSpaceModel()
        lsa_model = LSAModel()
        
        st.success("All models initialized successfully!")
        return vsm_model, lsa_model
        
    except Exception as e:
        st.error(f"Error initializing models: {str(e)}")
        return None, None

def display_results(results, documents, show_content=True):
    """Display search results"""
    if not results:
        st.warning("No results found")
        return
    
    st.subheader(f"Found {len(results)} Results:")
    
    for rank, result in enumerate(results, 1):
        doc_id = result['doc_id']
        score = result['score']
        source = result.get('source', 'Unknown')
        
        # Create result header
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"**#{rank} - Document {doc_id}**")
        with col2:
            st.markdown(f"Score: **{score:.4f}**")
        with col3:
            st.markdown(f"Model: **{source}**")
        
        if show_content and doc_id in documents:
            with st.expander(f"View Document {doc_id} Content"):
                content = documents[doc_id]
                # Truncate if too long
                if len(content) > 1000:
                    st.write(content[:1000] + "...")
                    st.caption("(Content truncated)")
                else:
                    st.write(content)
        
        st.divider()

# Streamlit App Configuration
st.set_page_config(
    page_title="Cranfield Search Engine Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Header
st.title("🔍 Cranfield Document Search Engine")
st.markdown("*Compare Vector Space Model and LSA with Boolean Search*")

# Load data and models
with st.spinner("Loading documents and initializing models..."):
    documents = load_cranfield_documents()
    queries = load_cranfield_queries()
    
    if documents:
        vsm_model, lsa_model = initialize_models(documents)
        
        if all([vsm_model, lsa_model]):
            st.success(f"✅ Loaded {len(documents)} documents and initialized all models")
        else:
            st.error("❌ Failed to initialize models")
            st.stop()
    else:
        st.error("❌ Failed to load documents")
        st.stop()

# Sidebar Configuration
with st.sidebar:
    st.header("🛠️ Search Configuration")
    
    # Model selection
    model_type = st.selectbox(
        "Select Search Model",
        ["Vector Space Model", "LSA + Boolean Search"],
        help="Choose the information retrieval model"
    )
    
    # Number of results
    top_k = st.slider(
        "Number of Results",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
        help="Maximum number of results to return"
    )
    
    # Show document content toggle
    show_content = st.checkbox("Show Document Content", value=True)
    
    st.divider()
    
    # Model information
    st.subheader("ℹ️ Model Information")
    if model_type == "Vector Space Model":
        st.info("**Vector Space Model**")
        st.markdown("""
        - Uses TF-IDF weighting
        - Cosine similarity matching
        - Traditional IR approach
        """)
    else:
        st.info("**LSA + Boolean Search**")
        st.markdown("""
        - Latent Semantic Analysis (SVD k=300)
        - Semantic similarity matching
        - Boolean operators support
        """)
        
        st.markdown("""
        **💡 Boolean Search Tips:**
        - `AND`: `flow AND pressure`
        - `OR`: `shock OR wave`  
        - `NOT`: `NOT turbulence`
        """)

# Main Content Area
st.header("🔎 Search Interface")

# Search type selection
search_type = st.radio(
    "Search Type",
    ["Free Text Search", "Predefined Queries"],
    horizontal=True,
    help="Free text: Enter your own query | Predefined: Use sample queries"
)

# Query input
if search_type == "Free Text Search":
    st.subheader("Enter Your Query")
    
    # Boolean search help for LSA
    if model_type == "LSA + Boolean Search":
        st.info("💡 You can use Boolean operators (AND, OR, NOT) with this model")
    
    query_text = st.text_area(
        "Query:",
        placeholder="Enter your search terms..." if model_type == "Vector Space Model"
        else "Enter search terms with optional Boolean operators (AND, OR, NOT)...",
        height=100
    )
    query_id = None
    
else:  # Predefined Queries
    st.subheader("Select Predefined Query")
    
    # Query selection with preview
    query_options = list(queries.items())
    selected_index = st.selectbox(
        "Available Queries:",
        range(len(query_options)),
        format_func=lambda i: f"Query {query_options[i][0]}: {query_options[i][1][:80]}..."
    )
    
    query_id, query_text = query_options[selected_index]
    
    # Display full query
    st.text_area("Selected Query:", value=query_text, height=100, disabled=True)

# Search execution
if st.button("🔍 Search", type="primary", use_container_width=True):
    if query_text.strip():
        with st.spinner("Searching..."):
            try:
                # Select model
                if model_type == "Vector Space Model":
                    model = vsm_model
                else:
                    model = lsa_model
                
                # Execute search
                start_time = datetime.now()
                results = model.query(query_text, top_k=top_k)
                search_time = (datetime.now() - start_time).total_seconds()
                
                # Store results in session state
                st.session_state.last_results = results
                st.session_state.last_query_text = query_text
                st.session_state.last_model = model_type
                st.session_state.search_time = search_time
                
                st.success(f"Search completed in {search_time:.3f} seconds")
                
            except Exception as e:
                st.error(f"Search failed: {str(e)}")
    else:
        st.warning("Please enter a search query")

# Display Results
if hasattr(st.session_state, 'last_results') and st.session_state.last_results:
    st.header("📋 Search Results")
    
    # Results summary
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"""
        **Query:** `{st.session_state.last_query_text}`  
        **Model:** {st.session_state.last_model}  
        **Results:** {len(st.session_state.last_results)} documents found
        """)
    with col2:
        st.metric("Search Time", f"{st.session_state.search_time:.3f}s")
    
    st.divider()
    display_results(st.session_state.last_results, documents, show_content)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <h4>About This Demo</h4>
    <p>This search engine compares two information retrieval approaches:</p>
    <div style='display: flex; justify-content: center; gap: 40px; margin: 20px 0;'>
        <div>
            <h5>Vector Space Model</h5>
            <ul style='text-align: left;'>
                <li>TF-IDF term weighting</li>
                <li>Cosine similarity</li>
                <li>Traditional approach</li>
            </ul>
        </div>
        <div>
            <h5>LSA + Boolean</h5>
            <ul style='text-align: left;'>
                <li>Semantic analysis (SVD)</li>
                <li>Boolean operators</li>
                <li>Advanced matching</li>
            </ul>
        </div>
    </div>
    <p><strong>Dataset:</strong> Cranfield Collection (1,400 aerodynamics documents)</p>
</div>
""", unsafe_allow_html=True)