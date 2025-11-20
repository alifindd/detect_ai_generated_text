import streamlit as st
import joblib
from src.preprocess_daigt import Cleaner, FeatureExtractor
import src.utils as utils
import pandas as pd
from docx import Document
import pymupdf

st.markdown("""
<style>
html, body, [class*="css"]  {
    height: 100%;
}

.main {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}

.info-dev {
    margin-bottom: 50px;
    color: #555;
    font-size: 14px;
    margin-top: -20px;
}
            
.divider {
    margin-top: auto;
    text-align: center;
    padding: 15px 0;
    color: #555;
    font-size: 14px;
            
</style>
""", unsafe_allow_html=True)

cleaner = Cleaner()
extractor = FeatureExtractor()

lr_model = joblib.load("models/pipeline_lr.joblib")
svm_model = joblib.load("models/pipeline_svc.joblib")
nb_model = joblib.load("models/pipeline_nb.joblib")
rf_model = joblib.load("models/pipeline_rf.joblib")

@st.dialog("Important Notice")
def show_notice():
    st.write("This detector can only well perform in english text, because its trained in english documents. Although you can try with another languages also.")

if "shown_popup" not in st.session_state:
    st.session_state.shown_popup = True
    show_notice()

st.title("AI Generated Text Detector")
st.markdown("""
<div class="info-dev">
    Developed by <b>alifindd</b> — 
    <a href='https://github.com/alifindd' target='_blank'>GitHub</a>
</div>
""", unsafe_allow_html=True)


uploaded_file = st.file_uploader("Upload file", type=["csv","docx","pdf"])
st.markdown("""
<div style="
    display: flex; 
    align-items: center; 
    text-align: center; 
">
    <hr style="flex: 1; border: none; border-top: 1px solid #ccc;">
    <span style="padding: 0 10px; color: #777; font-weight:600;">OR</span>
    <hr style="flex: 1; border: none; border-top: 1px solid #ccc;">
</div>
""", unsafe_allow_html=True)

user_input = st.text_area("Write text to predict:", height=200)


select_clf = st.selectbox("Select Classifier:", options=["LogisticRegression","SVM","Naive Bayes", "Random Forest"])
clfs = {
    "LogisticRegression" : lr_model,
    "SVM" : svm_model,
    "Naive Bayes" : nb_model,
    "Random Forest" : rf_model
}
selected_clf = clfs[select_clf]


if st.button("Predict"):
    if isinstance(user_input, str) and user_input.strip() != "":
        df = pd.DataFrame({"text": [user_input]})
        preds = selected_clf.predict(df)

        st.subheader("Prediction Result:")
        st.write("This text is AI Generated" if preds[0] == 1 else "This text is Human Written")

        if hasattr(selected_clf, "predict_proba"):
            proba = selected_clf.predict_proba(df)
            percent = int(proba[0][preds[0]] * 100)
            st.write("Probability:", f"{percent}%")
        else:
            st.write("Probability is not supported for this classifier")

        st.write("Model Used:", f"**{select_clf}**")



    elif uploaded_file is not None:
        def extract_docx(file):
            doc = Document(file)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            joined_text  = "\n".join(full_text)
            df = pd.DataFrame({"text": [joined_text]})
            return df
        
        def extract_pdf(file):
            pdf_bytes = file.read()
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            text_pdf = ""

            for page in doc:
                text_pdf += page.get_text()
        
            df = pd.DataFrame({"text": [text_pdf]})
            return df
        
        def extract_file(file, filename):
            if filename.endswith(".docx"):
                return extract_docx(file)
            elif filename.endswith(".csv"):
                return pd.read_csv(uploaded_file)
            elif filename.endswith(".pdf"):
                return extract_pdf(file)

        df = extract_file(uploaded_file, uploaded_file.name)
        df["word_length"] = df["text"].apply(utils.word_length)
        df["avg_sentence_length"] = df["text"].apply(utils.avg_sentence_length)
        df["punct_ratio"] = df["text"].apply(utils.punctuation_ratio)
        df["stopword_ratio"] = df["text"].apply(utils.stopword_ratio)

        if not uploaded_file.name.endswith(".csv"):
            preds = selected_clf.predict(df)
            st.subheader("Prediction Result:")
            st.write("This document is AI Generated" if preds[0] == 1 else "This document is Human Written")

            if hasattr(selected_clf, "predict_proba"):
                proba = selected_clf.predict_proba(df)
                percent = int(proba[0][preds[0]] * 100)
                st.write("Probability:", f"{percent}%")
            else:
                st.write("Probability is not supported for this classifier")
        else:
            preds = selected_clf.predict(df)
            label = ["Human" if x == 0 else "AI" for x in preds]
            df["label"] = label
            df_result = df[["text", "label"]]
            
            st.write("Showing first 5 rows of data")
            st.write(df_result.head())
        st.write("Model Used:", f"**{select_clf}**")
        
        


