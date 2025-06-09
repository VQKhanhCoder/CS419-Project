# Cranfield Document Search Engine

## 📖 Giới thiệu dự án

Cranfield Document Search Engine là một hệ thống tìm kiếm tài liệu sử dụng bộ dữ liệu Cranfield Collection. Dự án so sánh hiệu suất của ba mô hình Information Retrieval khác nhau:

- **Vector Space Model (VSM)**: Sử dụng TF-IDF và cosine similarity
- **Latent Semantic Analysis (LSA)**: Áp dụng SVD với k=300 dimensions và hỗ trợ Boolean search
- **Whoosh Search Engine**: Sử dụng thư viện Whoosh để full-text search

### 🎯 Mục tiêu

- So sánh hiệu suất các mô hình IR trên Cranfield dataset
- Đánh giá theo chuẩn TREC metrics (MAP, Precision, Recall)
- Cung cấp giao diện web thân thiện cho người dùng
- Hỗ trợ cả free text search và predefined queries

### 📊 Kết quả đánh giá

| Model | MAP Score | Precision | Recall |
|-------|-----------|-----------|---------|
| LSA Model | 0.178 | 0.191 | 0.322 |
| Vector Space Model | 0.158 | 0.182 | 0.308 |
| Whoosh Model | 0.004 | 0.009 | 0.005 |

## 📁 Cấu trúc dự án

```
CS419-Project/
├── data/                      # Dataset và dữ liệu
│   ├── Cranfield/            # 1,400 documents (.txt files)
│   ├── TEST/                 # Test queries & relevance judgments
│   │   ├── query.txt         # 225 test queries
│   │   └── RES/              # Relevance judgments per query
│   ├── Indexing/             # Generated inverted index
│   │   └── inverted_index.json
│   └── WhooshIndex/          # Whoosh search index
├── src/
│   ├── models/               # Search model implementations
│   │   ├── vector_space.py   # Vector Space Model với TF-IDF
│   │   ├── lsa_model.py      # LSA Model với SVD + Boolean search
│   │   └── whoosh_model.py   # Whoosh search integration
│   ├── preprocessing/        # Text processing modules
│   │   ├── preprocessing.py  # Main preprocessing pipeline
│   │   ├── build_vocab.py    # Vocabulary builder
│   │   └── vocabulary.json   # Generated vocabulary
│   ├── indexing/            # Index building
│   │   └── inverted_index.py # Inverted index creator
│   └── evaluation/          # Evaluation system
│       ├── evaluator.py     # TREC metrics evaluator
│       └── results/         # Evaluation results
├── web/                     # Web interface
│   └── web.py              # Streamlit web application
├── requirements.txt         # Dependencies
├── .gitignore              # Git ignore rules
├── run_web.py              # Web app entry point
└── README.md
```

### 🔧 Các thành phần chính

#### 1. Models (`src/models/`)
- **Vector Space Model**: TF-IDF weighting + cosine similarity
- **LSA Model**: Latent Semantic Analysis với SVD reduction + Boolean operators
- **Whoosh Model**: Full-text search engine integration

#### 2. Preprocessing (`src/preprocessing/`)
- Text normalization và tokenization với spaCy
- Stop words removal và lemmatization
- Vocabulary building từ corpus
- Hỗ trợ cả documents và queries

#### 3. Indexing (`src/indexing/`)
- Inverted index với position information
- Efficient storage và retrieval
- Support cho multiple search models

#### 4. Evaluation (`src/evaluation/`)
- TREC-style evaluation metrics
- Mean Average Precision (MAP)
- Precision-Recall curves
- Per-query performance analysis

#### 5. Web Interface (`web/`)
- Interactive Streamlit application
- Model comparison interface
- Real-time search và results display
- Support cho Boolean search operators

## 🚀 Cách chạy chương trình

### Bước 1: Cài đặt môi trường

```bash
# Clone repository
git clone <repository-url>
cd CS419-Project

# Tạo virtual environment (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### Bước 2: Download NLTK data và spaCy model

```bash
# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

# Download spaCy English model (this will download ~500MB)
python -m spacy download en_core_web_lg

# Note: Models are downloaded locally and excluded from Git repository
# This prevents large file issues with GitHub
```

### ⚠️ Important Notes

- **Model Files**: Large ML models (spaCy, NLTK data) are downloaded during setup and excluded from Git
- **Data Files**: Cranfield dataset should be placed in `data/` directory locally  
- **Generated Files**: Indices and caches are created locally and not tracked in Git
- **Cache Directories**: All model caches (transformers, spaCy, etc.) are automatically excluded

### 🗂️ Local File Structure After Setup

```
CS419-Project/
├── data/
│   ├── Cranfield/           # Place your documents here
│   └── TEST/               # Place your test queries here
├── .cache/                 # Generated caches (ignored by Git)
├── models/                 # Downloaded models (ignored by Git)
└── [other project files]
```

### Bước 3: Chuẩn bị dữ liệu

```bash
# Build vocabulary từ Cranfield documents
cd src/preprocessing
python build_vocab.py

# Tạo inverted index
cd ../indexing
python inverted_index.py
```

### Bước 4: Chạy web application

```bash
# Từ project root directory
python run_web.py

# Hoặc trực tiếp với Streamlit
streamlit run web/web.py
```

Mở browser và truy cập: `http://localhost:8501`

### Bước 5: Chạy evaluation (tùy chọn)

```bash
# Đánh giá tất cả models trên test queries
cd src/evaluation
python evaluator.py
```

## 🖥️ Sử dụng Web Interface

### 1. Model Selection
- **Vector Space Model**: Traditional TF-IDF approach
- **LSA + Boolean Search**: Semantic search với Boolean operators

### 2. Search Types
- **Free Text Search**: Nhập query tự do
- **Predefined Queries**: Chọn từ 225 test queries có sẵn

### 3. Boolean Search (LSA Model)
```
flow AND pressure          # Tìm documents chứa cả "flow" và "pressure"
shock OR wave              # Tìm documents chứa "shock" hoặc "wave"
boundary NOT layer         # Tìm documents chứa "boundary" nhưng không chứa "layer"
(heat AND transfer) OR flow # Combination với parentheses
```

### 4. Configuration Options
- Number of results (5-50)
- Show/hide document content
- Model-specific parameters

## 📈 Performance & Features

### ✅ Supported Features
- Multiple IR models comparison
- Boolean search operators (LSA model)
- Real-time search results
- TREC-standard evaluation metrics
- Interactive web interface
- Scalable architecture

### 🔍 Search Capabilities
- **Free text queries**: Natural language input
- **Sample queries**: 225 predefined test queries
- **Boolean operations**: AND, OR, NOT với LSA model
- **Semantic matching**: Context-aware search với LSA
- **Relevance scoring**: Confidence scores cho mỗi result

### 📊 Evaluation Metrics
- **Mean Average Precision (MAP)**
- **Precision at different recall levels**
- **11-point interpolated precision**
- **Per-query performance analysis**

## 🛠️ Development & Testing

### Chạy individual components:

```bash
# Test Vector Space Model
cd src/models
python vector_space.py

# Test LSA Model
python lsa_model.py

# Test Whoosh Model
python whoosh_model.py

# Test preprocessing
cd ../preprocessing
python preprocessing.py
```

### Debug và troubleshooting:
- Check logs trong console output
- Verify data files trong `data/` directory
- Ensure all dependencies được cài đặt đúng
- Kiểm tra file paths và permissions

## 📝 Dataset Information

**Cranfield Collection**:
- 1,400 aerodynamics documents
- 225 test queries
- Ground truth relevance judgments
- Standard IR evaluation benchmark

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Run tests
5. Submit pull request

## 📄 License

This project is for educational purposes in CS419 Information Retrieval course.
