#  MediMine🩺 — AI-Powered Medical Diagnosis System

Check out the **live Streamlit demo** here: [MediMine Application 🩺](https://medimine-application-bg7bcw2ukmizfkejnaxhxg.streamlit.app/)

## Project Overview
AI-powered system for predicting the most likely diseases and generating personalized recommendations using text similarity & embeddings.
This system scrapes medical data from NHS Inform, processes it using various ML models (Clustering, BiLSTM, BioBERT), and provides diagnosis predictions through a Flask API.
![App UI](assets/Screenshot 2025-11-14 180838.png)


## Features

- **Data Acquisition**: Web scraping from NHS Inform A-Z conditions
- **Multiple Models**: 
  - Clustering analysis 
  - BiLSTM neural network 
  - BioBERT transformer model 
- **RESTful API**: Flask-based endpoints for scraping and predictions
- **MongoDB Integration**: For data storage and model management

## Models Implemented

**Clustering Model**
- **Implemented by: Naira Ahmed**
- Approach: Agglomerative Clustering for symptom pattern discovery

**BiLSTM Model**
- **Implemented by: Teammate 1**
- Architecture: Bidirectional LSTM for symptom classification

**BioBERT Model**
- **Implemented by: Teammate 2**
- Fine-tuned BioBERT for medical text classification

## Prerequisites

- Python 3.8+
- MongoDB
- Required packages (see requirements.txt)

