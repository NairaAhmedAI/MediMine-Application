from flask import Flask, request, jsonify
from pymongo import MongoClient
import gridfs
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import json
import os

# 1️⃣ Initialize Flask app
app = Flask(__name__)

# 2️⃣ Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["NHS_DB"]
conditions_col = db["conditions"]
fs = gridfs.GridFS(db)
models_meta = db["models_meta"]

# مكان مؤقت لتخزين النموذج
MODEL_PATH = "bilstm_model.h5"
TOKENIZER_PATH = "tokenizer.json"

# 3️⃣ Endpoint لتدريب النموذج


@app.route("/train", methods=["POST"])
def train_model():
    # --- Step 1: جلب البيانات ---
    data = list(conditions_col.find({}))
    if not data:
        return jsonify({"error": "No data found in MongoDB"}), 400

    X = [d.get("symptoms", []) + d.get("warnings", []) +
         d.get("recommendations", []) for d in data]
    X = [" ".join(x) for x in X]  # دمج القوائم في نص
    all_conditions = [d["condition"] for d in data]

    # --- Step 2: تجهيز labels ---
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform([[c] for c in all_conditions])

    # --- Step 3: تقسيم البيانات ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # --- Step 4: Tokenization ---
    max_words = 30000
    max_len = 200
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    seq_train = pad_sequences(
        tokenizer.texts_to_sequences(X_train), maxlen=max_len)
    seq_test = pad_sequences(
        tokenizer.texts_to_sequences(X_test), maxlen=max_len)

    n_classes = y_train.shape[1]

    # --- Step 5: بناء BiLSTM ---
    inp = Input(shape=(max_len,))
    x = Embedding(input_dim=max_words, output_dim=128,
                  input_length=max_len)(inp)
    x = Bidirectional(LSTM(128))(x)
    x = Dropout(0.5)(x)
    out = Dense(n_classes, activation="sigmoid")(x)

    model = Model(inp, out)
    model.compile(loss="binary_crossentropy",
                  optimizer="adam", metrics=["AUC"])

    # --- Step 6: تدريب ---
    es = EarlyStopping(patience=3, restore_best_weights=True)
    model.fit(seq_train, y_train, validation_split=0.1,
              epochs=5, batch_size=64, callbacks=[es])

    # --- Step 7: تقييم ---
    y_pred_proba = model.predict(seq_test)
    y_pred_bin = (y_pred_proba >= 0.5).astype(int)

    def top_k_accuracy(y_true, y_pred, k=1):
        top_k = np.argsort(y_pred, axis=1)[:, -k:]
        matches = np.array([any(top_k[i] == np.where(y_true[i] == 1)[0])
                           for i in range(len(y_true))])
        return matches.mean()

    top1 = top_k_accuracy(y_test, y_pred_proba, k=1)
    top3 = top_k_accuracy(y_test, y_pred_proba, k=3)
    f1 = f1_score(y_test, y_pred_bin, average="micro")

    # --- Step 8: حفظ النموذج + tokenizer ---
    model.save(MODEL_PATH)
    with open(TOKENIZER_PATH, "w") as f:
        f.write(tokenizer.to_json())

    with open(MODEL_PATH, "rb") as f:
        model_id = fs.put(f, filename="bilstm_model.h5")
    with open(TOKENIZER_PATH, "rb") as f:
        tok_id = fs.put(f, filename="tokenizer.json")

    models_meta.delete_many({"name": "bilstm_model"})  # تحديث فقط آخر نسخة
    models_meta.insert_one({
        "name": "bilstm_model",
        "type": "BiLSTM",
        "gridfs_id": model_id,
        "tokenizer_id": tok_id,
        "labels": list(mlb.classes_),
        "metrics": {"Top-1": round(float(top1), 3), "Top-3": round(float(top3), 3), "Micro-F1": round(float(f1), 3)}
    })

    return jsonify({
        "message": "Model trained and saved successfully",
        "metrics": {"Top-1": top1, "Top-3": top3, "Micro-F1": f1}
    })

# 4️⃣ Endpoint للتنبؤ


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    symptoms = data.get("symptoms", [])

    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    # استرجاع النموذج والـtokenizer
    meta = models_meta.find_one({"name": "bilstm_model"})
    if not meta:
        return jsonify({"error": "Model not trained yet"}), 400

    model_file = fs.get(meta["gridfs_id"])
    tokenizer_file = fs.get(meta["tokenizer_id"])

    with open(MODEL_PATH, "wb") as f:
        f.write(model_file.read())
    with open(TOKENIZER_PATH, "wb") as f:
        f.write(tokenizer_file.read())

    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "r") as f:
        tokenizer = tokenizer_from_json(f.read())

    labels = meta["labels"]
    max_len = model.input_shape[1]

    # تجهيز النص
    input_text = " ".join(symptoms)
    seq = tokenizer.texts_to_sequences([input_text])
    seq_padded = pad_sequences(seq, maxlen=max_len)

    # التنبؤ
    y_pred_proba = model.predict(seq_padded)[0]
    top3_idx = y_pred_proba.argsort()[-3:][::-1]
    top3_results = [
        {"disease": labels[i], "probability": float(y_pred_proba[i])}
        for i in top3_idx
    ]

    return jsonify({
        "input_symptoms": symptoms,
        "predictions": top3_results
    })


# Run server
if __name__ == "__main__":
    app.run(debug=True, port=5000)
