#  MediMine🩺 — AI-Powered Medical Diagnosis System

Check out the **live Streamlit demo** here: [MediMine Application 🩺](https://medimine-application-bg7bcw2ukmizfkejnaxhxg.streamlit.app/)

## Project Overview
AI-powered system for predicting the most likely diseases and generating personalized recommendations using text similarity & embeddings.
This system scrapes medical data from NHS Inform, processes it using various ML models (Clustering, BiLSTM, BioBERT), and provides diagnosis predictions through a Flask API.

<img width="1738" height="1018" alt="image" src="https://github.com/user-attachments/assets/7cd0c978-7862-4240-b8d6-e35e7fc0cf12" />

## Features

- **Data Acquisition**: Web scraping from NHS Inform A-Z conditions
- **Multiple Models**: 
  - Clustering analysis 
  - BiLSTM neural network 
  - BioBERT transformer model
  - BERT transformer model
- **RESTful API**: Flask-based endpoints for scraping and predictions
- **MongoDB Integration**: For data storage and model management

## Models Implemented

**Clustering Model**
- **Implemented by: [Naira Ahmed](https://github.com/NairaAhmedAI)**
- Approach: Agglomerative Clustering for symptom pattern discovery

**BiLSTM Model**
- **Implemented by: [Yassmen abdelaziz](https://github.com/yasmenelqady)**
- Architecture: Bidirectional LSTM for symptom classification

**BioBERT and BERT Models (Transformers)**
- **Implemented by: [Basma Sameh](https://github.com/basmasameh84)**
- Fine-tuned BioBERT for medical text classification
- Fine-tuned BERT for medical text classification

# 🧹 Data Scraping Pipeline — NHS Inform A–Z Scraper

The system includes an automated web-scraping pipeline designed to collect medical content from [NHS Inform (A–Z condition](https://www.nhsinform.scot/)
and convert it into structured, machine-readable records stored in MongoDB.

## 🔍 Scraping Process Overview

### 1. Entry Point — A–Z Index:
 - The scraper begins at the official NHS A–Z index page and extracts all condition names and their URLs.

### 2. Content Extraction:
 - For each condition page, the scraper systematically collects:
     - Section headings `(<h2>)`.
     - Paragraphs `(<p>)`.
     - Bullet lists `(<ul><li>)`.
   
### 3. Intelligent Section Mapping:
  - A custom keyword-driven mapping engine categorizes extracted content into:
    - Symptoms.
    - Causes.
    - Diagnosis.
    - Warnings .
    - Recommendations .
   * This ensures consistent structuring even when individual NHS pages differ in layout.

### 4. MongoDB Insertion:
 - Each processed condition is stored as a structured document:
```python

 {
  "condition": "Asthma",
  "symptoms": "...",
  "causes": "...",
  "diagnosis": "...",
  "warnings": "...",
  "recommendations": "..."
}

 ```
# 🧬 Advanced Medical Text Preprocessing & Input/Output Construction
This module prepares scraped medical data for ML and Transformer models using ` SciSpacy ` and structured input/output creation.

## Key Steps & Highlights

### 1. Load SciSpacy Medical Model
  - Lightweight model en_core_sci_sm optimized for biomedical/scientific text.
  - Provides medical entity recognition and tokenization.

### 2. Define Advanced Medical Processing Function
- Input text is processed through the SciSpacy pipeline.
- Extracts medical entities: diseases, drugs, procedures.
- Extracts lemmatized tokens:
   - Ignores stopwords (e.g., “the”, “and”)
   - Keeps only alphabetic tokens (removes numbers, symbols).
   - Merges entities + tokens into a single clean string.

### 3. Apply Processing to All Text Fields
 - Fields processed: `symptoms`, `causes`, `diagnosis`, `warnings`, `recommendations `.
 - Ensures uniform cleaning and normalization for ML models.

### 4. Build Model Input/Output Pairs
 - Input (`input_text`): combination of `symptoms` + `causes`.
 - Handles missing values (some diseases lack symptoms, others lack causes).
 - Merging ensures all records contribute to model learning.
 - Output (`output_text`): structured text combining:
   
 ```python   
Disease: ... | Recommendations: ... | Warnings: ... | Diagnosis: ...

 ```
 - This format is suitable for `clustering`, `seq2seq`, or `Transformer` models.

### 5. Save Advanced Processed Dataset
- `diseases_advanced_processed.json` → cleaned medical fields.
- `diseases_final_for_model.json` → final input/output pairs ready for ML/Transformer training.

### 6.Result
  - Dataset is noise-free, medical-aware, and ready for training.
  - Improves performance for ML ,DNN and BERT/BioBERT models.
  - Preserves learning from incomplete records by merging available symptoms and causes.

# 🗂️ Agglomerative Clustering Model (Medical Conditions)

## Overview
 The model groups medical conditions into clusters based on textual similarity.
 Uses **TF-IDF vectorization** of condition names and **Agglomerative Clustering**.
 Stored in **MongoDB with GridFS** along with metadata for easy retrieval via API.

## Key Steps

### 1. Data Loading:  
 - Fetch conditions from MongoDB (`conditions` collection).
 - Convert to DataFrame for processing.

### 2. Feature Extraction
 - TF-IDF vectorization (`max_features=5000`, ngram_range=(1,2)) of condition names.

### 3. Agglomerative Clustering
 - Clusters: `n_clusters=60`
 - Metric: cosine similarity, linkage: average.
 - Assigns each condition to a cluster (`agg_cluster`).

### 4. Evaluation
 - Calculates **intra-cluster cohesion** using cosine similarity.
 - Provides a simple metric to estimate cluster quality.

### 5. Model & Metadata Storage
 - Model + TF-IDF vectorizer serialized with `pickle` and saved to **GridFS**.
 - Metadata stored in `models_meta1` collection:
 - Name, type, labels, cohesion score, creation timestamp.

### 6. API Usage
 - The Flask API loads this model from MongoDB/GridFS.
 - Provides endpoints to:
     - Retrieve clusters for a given condition.
     - Serve as a backend for `Streamlit` recommendations interface.

### 7. Result
  - Enables **text-based grouping of medical conditions**.
  - Supports **recommendation and similarity** search in medical applications.
  - Ready to integrate with Streamlit or other ML/AI interfaces.

# BiLSTM Medical Condition Classification API

This Flask API provides training and inference endpoints for a **BiLSTM-based multi-class medical condition classifier**.
All models, tokenizers, and metadata are stored securely in  **MongoDB (GridFS)** to enable versioning and dynamic deployment.

## Key Features

- **End-to-end ML pipeline** (training → evaluation → deployment).
- **BiLSTM neural network** for medical text classification.
- **MultiLabelBinarizer** for encoding condition labels.
- **Tokenizer + padding** to prepare text inputs for the model.
- **Model & tokenizer stored in GridFS**, enabling remote loading.
- **Top-1 / Top-3 accuracy + Micro-F**1 as evaluation metrics.
- **Fully functional** ّ/predictّ`**endpoint** for real-time predictions.
  
## 📌 Main Components
### 1. MongoDB Integration

- Reads medical data from ّconditionsّ collection.
- Stores trained models and tokenizers in **GridFS**.
- Saves model version metadata in ّmodels_metaّ.

### 2. Training Endpoint — /train
   
* This endpoint:
- 1.  **Loads training data** (symptoms + warnings + recommendations).
- 2.  **Preprocesses labels** using MultiLabelBinarizer.
- 3.  **Splits data** (80% training, 20% testing).
- 4. **Builds a BiLSTM model**:
      - Embedding layer (128-dim)
      - Bidirectional LSTM (128 units)
      - Dropout for regularization
      - Sigmoid output for multi-label classification
- 5. **Trains with EarlyStopping**
- 6. **Evaluates performance**:
       - Top-1 accuracy
       - Top-3 accuracy
       - Micro-F1 score
- 7. **Saves the model + tokenizer** to GridFS.
- 8. Updates model metadata with version, labels & metrics.

### 3. **Prediction Endpoint** — `/predict`

- Loads the latest BiLSTM model and tokenizer from GridFS.
- Processes incoming symptoms into padded sequences.
- Generates probabilities for all diseases.
- Returns **Top-3 predicted conditions** with confidence scores.

### 4. **API Workflow Summary**

1. **Client sends symptoms**
2. `/predict`→ loads model → tokenizes text
3. Model outputs probability distribution
4. API returns Top-3 conditions + probabilities

