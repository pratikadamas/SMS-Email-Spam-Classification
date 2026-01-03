import streamlit as st
import pickle
from text_utils import transform_text

# Load model & vectorizer
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Spam Classifier", page_icon="📩", layout="centered")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stTextArea textarea {
        border-radius: 10px;
    }
    .result-container {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
        font-weight: bold;
        font-size: 22px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }
    .spam-box {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #ef5350;
    }
    .ham-box {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 2px solid #66bb6a;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.markdown("<h1 style='text-align:center;'>📩 Spam  SMS/Email Classifier</h1>", unsafe_allow_html=True)
st.write("Detect spam messages across SMS, Email, and Social Media using Machine Learning.")
st.write("---")

# --- SESSION STATE ---
if "text" not in st.session_state:
    st.session_state.text = ""

# --- INPUT AREA ---
input_sms = st.text_area("Paste your message here:", height=150, value=st.session_state.text)

col1, col2 = st.columns([1, 1])

with col1:
    predict_btn = st.button("🔍 Analyze Message", use_container_width=True, type="primary")

with col2:
    if st.button("🧹 Clear Input", use_container_width=True):
        st.session_state.text = ""
        st.rerun()

# --- PREDICTION LOGIC ---
if predict_btn:
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message to analyze.")
    else:
        # Preprocess & Predict
        transformed = transform_text(input_sms)
        vector = tfidf.transform([transformed])
        result = model.predict(vector)[0]

        # Display Result in Styled Div
        if result == 1:
            st.markdown(
                f'<div class="result-container spam-box">🚨 DETECTED AS SPAM<br><small style="font-size: 14px;">This message contains patterns typical of fraud or phishing.</small></div>', 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-container ham-box">✅ MESSAGE IS SAFE<br><small style="font-size: 14px;">This looks like a legitimate conversation.</small></div>', 
                unsafe_allow_html=True
            )