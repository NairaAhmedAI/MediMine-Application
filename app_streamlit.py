import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity


with open("agglomerative_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


with open("recommendations.pkl", "rb") as f:
    recommendations = pickle.load(f)

# ---Streamlit ---
st.title("Medical Disease Predictor 🩺")

# Text Area 
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

      
        df = df.sort_values(by="Similarity", ascending=False)

      
        df['Recommendation'] = df['Disease'].apply(
            lambda x: recommendations.get(x, "No recommendation available"))

     
        st.subheader("Predicted Diseases with Similarity & Recommendations")
        st.dataframe(df)

       
        st.subheader("Top 4 Possible Diseases")
        st.table(df.head(5))

