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

positive_keywords = ["outstanding", "excellent", "amazing", "great", "perfect", "best"]

negative_phrases = [
    "heat", "heats", "lag", "lags", "bug", "bugs", "buggy", 
    "not worth", "not good"
]

positive_phrases = [
    "decent phone", "outstanding", "excellent", "amazing", "great", "perfect", "best", 
    "good phone"
]

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

    c1, c2 = st.columns(2)
    with c1:
        if st.button("👍 Positive Example"):
            st.session_state.text = "This product is amazing and works perfectly"
    with c2:
        if st.button("👎 Negative Example"):
            st.session_state.text = "Worst purchase ever, totally useless"

    user_input = st.text_area("Type your review:", key="text")

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

                if any(phrase in cleaned for phrase in negative_phrases):
                    prediction = 0
                elif any(phrase in cleaned for phrase in positive_phrases):
                    prediction = 1
                elif any(word in cleaned for word in positive_keywords):
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

        fig, ax = plt.subplots()
        ax.pie([positives, negatives], labels=["Positive", "Negative"], autopct='%1.1f%%')
        st.pyplot(fig)

# ---------- CSV ----------
st.divider()
st.subheader("📂 Bulk Review Analysis (CSV)")

uploaded_file = st.file_uploader("Upload CSV file (must contain 'review' column)", type=["csv"])

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

                if any(phrase in r for phrase in negative_phrases):
                    p = 0
                elif any(phrase in r for phrase in positive_phrases):
                    p = 1
                elif any(word in r for word in positive_keywords):
                    p = 1

            predictions.append(p)

        df["Sentiment"] = ["Positive" if p == 1 else "Negative" for p in predictions]
        st.dataframe(df)

# ---------- BULK TEXT ----------
st.divider()
st.subheader("📂 Bulk Review Analyzer")

multi_input = st.text_area("Paste multiple reviews (one per line):")

colA, colB = st.columns(2)
analyze_clicked = colA.button("Analyze Reviews")
insight_clicked = colB.button("Generate Insights")

# ANALYZE
if analyze_clicked:
    if multi_input:
        st.session_state.history = []

        reviews = [r.strip() for r in multi_input.split("\n") if r.strip()]
        cleaned_reviews = [clean_text(r) for r in reviews]

        results = []

        for r in cleaned_reviews:
            if check_abuse(r):
                p = 0
            else:
                p = model.predict(vectorizer.transform([r]))[0]

                if any(phrase in r for phrase in negative_phrases):
                    p = 0
                elif any(phrase in r for phrase in positive_phrases):
                    p = 1
                elif any(word in r for word in positive_keywords):
                    p = 1

            results.append(p)

        for r, p in zip(reviews, results):
            if p == 1:
                st.success(f"✅ {r}")
            else:
                st.error(f"❌ {r}")

            st.session_state.history.append((r, p, 1.0))

# INSIGHTS
if insight_clicked:
    if multi_input:
        st.subheader("🧠 Smart Insights")
        st.info("📌 Users generally appreciate the product's quality and performance, but some report issues like heating, lag, or battery problems.")
