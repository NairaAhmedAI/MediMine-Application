# 🧠 BiLSTM Medical Condition Classifier API

This project provides a **Flask-based API** that trains a **BiLSTM deep learning model** on medical text data (symptoms, warnings, recommendations) and predicts the most likely medical conditions.

---

## 📌 Features
- **Train endpoint** → trains a BiLSTM model using MongoDB data.
- **Predict endpoint** → returns the Top‑3 predicted diseases.
- **MongoDB + GridFS integration** → store model + tokenizer.
- **Multi‑label encoding** using `MultiLabelBinarizer`.
- **Deep Learning** using Bidirectional LSTM.

---

## 🛠️ Technologies Used
- **Python**
- **Flask** (API)
- **TensorFlow / Keras** (BiLSTM model)
- **MongoDB + GridFS**
- **Scikit‑learn**
- **NumPy**

---

## 📂 Project Structure
```
project/
│   bilstm2_fapi.py
│   requirements.txt
│   README.md
└── models/
```

---

## 🚀 How to Run the Project

### 1️⃣ Install dependencies
```
pip install -r requirements.txt
```

### 2️⃣ Start MongoDB
Make sure MongoDB server is running locally:
```
mongod
```

### 3️⃣ Run the Flask API
```
python bilstm2_fapi.py
```
The API will start on:
```
http://127.0.0.1:5000
```

---

## 📌 API Endpoints

### 🔹 1. Train Model
**POST** `/train`

Trains the BiLSTM model using data from MongoDB.

**Response example:**
```json
{
  "message": "Model trained and saved successfully",
  "metrics": {
    "Top-1": 0.62,
    "Top-3": 0.81,
    "Micro-F1": 0.55
  }
}
```

---

### 🔹 2. Predict Disease
**POST** `/predict`

#### Request body:
```json
{
  "symptoms": ["fever", "cough", "fatigue"]
}
```

#### Response example:
```json
{
  "input_symptoms": ["fever", "cough", "fatigue"],
  "predictions": [
    {"disease": "Influenza", "probability": 0.85},
    {"disease": "Common Cold", "probability": 0.73},
    {"disease": "Pneumonia", "probability": 0.41}
  ]
}
```

---

## 🧬 Model Details
- **Embedding size:** 128
- **BiLSTM units:** 128
- **Dropout:** 0.5
- **Loss:** Binary Crossentropy
- **Optimizer:** Adam
- **Max sequence length:** 200
- **Vocabulary size:** 30,000

---

## 🗂️ Stored in MongoDB
The API stores:
- BiLSTM model → GridFS
- tokenizer → GridFS
- labels + metrics → `models_meta` collection

---

## 👥 Authors
Developed for a medical diagnosis project using NLP + Deep Learning.

---

## 📄 License
This project is for educational and research purposes.

