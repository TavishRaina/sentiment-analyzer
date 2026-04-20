import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import re

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------- TEXT CLEANING ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

# ---------- KEYWORD FIX ----------
positive_keywords = ["outstanding", "excellent", "amazing", "great", "perfect", "best"]

# ---------- ABUSE FILTER ----------
abusive_words = [
    "fuck", "fucking", "fuckable",
    "shit", "dogshit", "bullshit",
    "bitch", "motherfucker", "mf", "kutta", "sala", "saala", "bhadwa", "bhdwa", "chutiya", "randwa", "randi",
    "sisterfucker",
    "madarchod", "madar chod",
    "maa ki chut", "maa ke lwde",
    "behen ki chut",
    "bhosdk", "bkl", "bkc", "mkc",
    "bc", "mc",
    "lwde", "lawde", "lund",
    "gandu", "gaandu",
    "trash"
]

def check_abuse(text):
    for word in abusive_words:
        if word in text:
            return True
    return False

# Page config
st.set_page_config(page_title="AI Sentiment Analyzer", layout="wide")

# ---------- HEADER ----------
st.markdown("<h1 style='text-align:center;'>🧠 AI Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Analyze product reviews using Machine Learning</p>", unsafe_allow_html=True)

st.divider()

# ---------- SESSION ----------
if "history" not in st.session_state:
    st.session_state.history = []

col1, col2 = st.columns([2, 1])

# ---------- LEFT ----------
with col1:
    st.subheader("✍️ Enter Review")

    user_input = st.text_area("Type your review:")

    if st.button("🚀 Predict Sentiment"):
        if user_input:
            cleaned = clean_text(user_input)

            if check_abuse(cleaned):
                prediction = 0
                prob = 1.0
            else:
                data = vectorizer.transform([cleaned])
                prediction = model.predict(data)[0]
                prob = model.predict_proba(data)[0][prediction]

                # KEYWORD FIX
                if any(word in cleaned for word in positive_keywords):
                    prediction = 1

            st.session_state.history.append((user_input, prediction, prob))

            if prediction == 1:
                st.success("✅ Positive Review")
            else:
                st.error("❌ Negative Review")

            st.progress(prob)

# ---------- RIGHT ----------
with col2:
    st.subheader("📊 Stats")

    if st.session_state.history:
        total = len(st.session_state.history)
        positives = sum(1 for x in st.session_state.history if x[1] == 1)
        negatives = total - positives

        st.metric("Total", total)
        st.metric("Positive", positives)
        st.metric("Negative", negatives)

# ---------- CSV ----------
st.divider()
st.subheader("📂 Bulk Review Analysis")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "review" in df.columns:
        cleaned_reviews = df["review"].apply(clean_text)

        predictions = []
        for r in cleaned_reviews:
            if check_abuse(r):
                p = 0
            else:
                p = model.predict(vectorizer.transform([r]))[0]

                # KEYWORD FIX
                if any(word in r for word in positive_keywords):
                    p = 1

            predictions.append(p)

        df["Sentiment"] = ["Positive" if p == 1 else "Negative" for p in predictions]

        # SHOW ALL REVIEWS
        st.dataframe(df)

        st.success(f"Positive: {sum(predictions)} | Negative: {len(predictions)-sum(predictions)}")

# ---------- BULK INPUT ----------
st.divider()
st.subheader("📂 Bulk Review Analyzer")

multi_input = st.text_area("Paste multiple reviews (one per line):")

if st.button("Analyze Reviews"):
    if multi_input:
        reviews = [clean_text(r.strip()) for r in multi_input.split("\n") if r.strip()]

        for r in reviews:
            if check_abuse(r):
                st.error(f"❌ {r}")
            else:
                p = model.predict(vectorizer.transform([r]))[0]

                # KEYWORD FIX
                if any(word in r for word in positive_keywords):
                    p = 1

                if p == 1:
                    st.success(f"✅ {r}")
                else:
                    st.error(f"❌ {r}")

# ---------- HISTORY ----------
st.divider()
st.subheader("📜 History")

for review, pred, _ in st.session_state.history[-5:]:
    if pred == 1:
        st.success(review)
    else:
        st.error(review)

# ---------- FOOTER ----------
st.divider()
st.markdown("<p style='text-align:center;'>Built using Machine Learning + Streamlit</p>", unsafe_allow_html=True)
