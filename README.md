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
- **Implemented by: Naira Ahmed**
- Approach: Agglomerative Clustering for symptom pattern discovery

**BiLSTM Model**
- **Implemented by: Yassmen abdelaziz**
- Architecture: Bidirectional LSTM for symptom classification

**BioBERT and BERT Models (Transformers)**
- **Implemented by: Basma Sameh**
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
    - Warnings / Emergency .
    - Recommendations .
This ensures consistent structuring even when individual NHS pages differ in layout.

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


