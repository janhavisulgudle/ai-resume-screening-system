import streamlit as st
import pickle

model = pickle.load(open("model/resume_classifier.pkl","rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl","rb"))

st.title("AI Resume Screening System")

resume = st.text_area("Paste Resume Text")

if st.button("Analyze Resume"):
    text = vectorizer.transform([resume])
    prediction = model.predict(text)
    st.write("Predicted Job Category:", prediction[0])
