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
1. Entry Point — A–Z Index:
 - The scraper begins at the official NHS A–Z index page and extracts all condition names and their URLs.

3. Content Extraction:
 - For each condition page, the scraper systematically collects:
     - Section headings `(<h2>)`.
     - Paragraphs `(<p>)`.
     - Bullet lists `(<ul><li>)`.
   
4. Intelligent Section Mapping:
  - A custom keyword-driven mapping engine categorizes extracted content into:
    - Symptoms.
    - Causes.
    - Diagnosis.
    - Warnings .
    - Recommendations .
   * This ensures consistent structuring even when individual NHS pages differ in layout.

6. MongoDB Insertion:
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

1. Load SciSpacy Medical Model
  - Lightweight model en_core_sci_sm optimized for biomedical/scientific text.
  - Provides medical entity recognition and tokenization.

2. Define Advanced Medical Processing Function
- Input text is processed through the SciSpacy pipeline.
- Extracts medical entities: diseases, drugs, procedures.
- Extracts lemmatized tokens:
   - Ignores stopwords (e.g., “the”, “and”)
   - Keeps only alphabetic tokens (removes numbers, symbols).
   - Merges entities + tokens into a single clean string.

3. Apply Processing to All Text Fields
 - Fields processed: symptoms, causes, diagnosis, warnings, recommendations.
 - Ensures uniform cleaning and normalization for ML models.

4. Build Model Input/Output Pairs
 - Input (input_text): combination of `symptoms` + `causes`.
 - Handles missing values (some diseases lack symptoms, others lack causes).
 - Merging ensures all records contribute to model learning.
 - Output (output_text): structured text combining:
   
 ```python   
Disease: ... | Recommendations: ... | Warnings: ... | Diagnosis: ...

 ```
 - This format is suitable for classification, seq2seq, or Transformer models.

5. Save Advanced Processed Dataset
- `diseases_advanced_processed.json` → cleaned medical fields.
- `diseases_final_for_model.json` → final input/output pairs ready for ML/Transformer training.

 6.Result
  - Dataset is noise-free, medical-aware, and ready for training.
  - Improves performance for ML ,DNN and BERT/BioBERT models.
  - Preserves learning from incomplete records by merging available symptoms and causes.

