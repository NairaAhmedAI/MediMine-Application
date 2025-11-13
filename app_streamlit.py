import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# --- تحميل الموديل والفكتورايزر ---
with open("agglomerative_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# لو عندك توصيات محفوظة
with open("recommendations.pkl", "rb") as f:
    recommendations = pickle.load(f)

# --- واجهة Streamlit ---
st.title("Medical Disease Predictor 🩺")

# Text Area لإدخال الأعراض
user_input = st.text_area("Enter your symptoms (separate by commas):")

# زر Predict
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter your symptoms first!")
    else:
        # تحويل الأعراض لتمثيل TF-IDF
        user_vec = vectorizer.transform([user_input])

        # مصفوفة أعراض الأمراض من الموديل
        disease_vecs = vectorizer.transform(model['disease_symptoms'])

        # حساب التشابه
        similarity = cosine_similarity(user_vec, disease_vecs)[0]

        # إنشاء DataFrame للنتائج
        df = pd.DataFrame({
            "Disease": model['diseases'],
            "Similarity": similarity
        })

        # ترتيب من الأعلى للأسفل
        df = df.sort_values(by="Similarity", ascending=False)

        # إضافة التوصيات
        df['Recommendation'] = df['Disease'].apply(
            lambda x: recommendations.get(x, "No recommendation available"))

        # عرض الجدول الكامل
        st.subheader("Predicted Diseases with Similarity & Recommendations")
        st.dataframe(df)

        # عرض أفضل 3 نتائج
        st.subheader("Top 3 Possible Diseases")
        st.table(df.head(5))
