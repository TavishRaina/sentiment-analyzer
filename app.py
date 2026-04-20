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
    text = text.lower()
    for word in abusive_words:
        if word in text:
            return True
    return False

# Page config
st.set_page_config(
    page_title="AI Sentiment Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- UI STYLE ----------
st.markdown("""
<style>
html, body, .stApp {
    background-color: #f8fafc !important;
    color: #1e293b !important;
}
h1 {
    text-align: center !important;
    color: #1e293b !important;
}
h2, h3 {
    text-align: left !important;
    color: #1e293b !important;
}
.block {
    padding: 20px;
    border-radius: 10px;
    background: white !important;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}
textarea, input {
    background-color: white !important;
    color: #1e293b !important;
}
.stButton>button {
    background: linear-gradient(135deg, #6366f1, #3b82f6);
    color: white !important;
    border-radius: 8px;
    border: none;
}
* {
    color: #1e293b !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("<h1>🧠 AI Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Analyze product reviews using Machine Learning</p>", unsafe_allow_html=True)

st.divider()

# ---------- SESSION STATE ----------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- LAYOUT ----------
col1, col2 = st.columns([2, 1])

# ---------- LEFT PANEL ----------
with col1:
    st.markdown("<div class='block'>", unsafe_allow_html=True)

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
            if any(word in cleaned for word in positive_keywords):
                prediction = 1
    
            st.session_state.history.append((user_input, prediction, prob))

            st.subheader("🔍 Result")

            if prediction == 1:
                st.success("✅ Positive Review")
            else:
                st.error("❌ Negative Review")

            st.progress(prob)
            st.write(f"Confidence Score: *{prob:.2f}*")

            stars = int(prob * 5)
            st.write("⭐" * stars)

        else:
            st.warning("Please enter a review")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- RIGHT PANEL ----------
with col2:
    st.markdown("<div class='block'>", unsafe_allow_html=True)

    st.subheader("📊 Model Info")
    st.write("Algorithm: Logistic Regression")
    st.write("Technique: TF-IDF")
    st.write("Accuracy: ~87%")

    st.subheader("📈 Stats")

    if st.session_state.history:
        total = len(st.session_state.history)
        positives = sum(1 for x in st.session_state.history if x[1] == 1)
        negatives = total - positives

        st.metric("Total Predictions", total)
        st.metric("Positive", positives)
        st.metric("Negative", negatives)

        fig, ax = plt.subplots()
        ax.pie([positives, negatives], labels=["Positive", "Negative"], autopct='%1.1f%%')
        st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- CSV UPLOAD ----------
st.divider()
st.subheader("📂 Bulk Review Analysis")

uploaded_file = st.file_uploader("Upload CSV file (must contain 'review' column)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "review" in df.columns:
        cleaned_reviews = df["review"].apply(clean_text)

        predictions = []
        for r in cleaned_reviews:
            if check_abuse(r):
                predictions.append(0)
            else:
                p = model.predict(vectorizer.transform([r]))[0]
                if any(word in r for word in positive_keywords):
                   p = 1
                predictions.append(p)

        df["Sentiment"] = ["Positive" if p == 1 else "Negative" for p in predictions]

        st.dataframe(df)

        pos = sum(predictions)
        neg = len(predictions) - pos

        st.success(f"Positive: {pos} | Negative: {neg}")
    else:
        st.error("CSV must contain 'review' column")

# ---------- BULK ANALYZER ----------
st.divider()
st.subheader("📂 Bulk Review Analyzer")

multi_input = st.text_area("Paste multiple reviews (one per line):", height=200)

if st.button("Analyze Reviews"):
    if multi_input:
        reviews = [clean_text(r.strip()) for r in multi_input.split("\n") if r.strip()]

        st.subheader("🔍 Results")

        for r in reviews:
            if check_abuse(r):
                st.error(f"❌ {r}")
            else:
                p = model.predict(vectorizer.transform([r]))[0]
                if p == 1:
                    st.success(f"✅ {r}")
                else:
                    st.error(f"❌ {r}")
    else:
        st.warning("Please enter reviews")

# ---------- HISTORY ----------
st.divider()
st.subheader("📜 Prediction History")

if st.session_state.history:
    for review, pred, prob in reversed(st.session_state.history[-5:]):
        if pred == 1:
            st.success(f"✅ {review[:80]}... ({prob:.2f})")
        else:
            st.error(f"❌ {review[:80]}... ({prob:.2f})")
else:
    st.info("No predictions yet")

# ---------- FOOTER ----------
st.divider()
st.markdown("<p style='text-align:center;'>Built using Machine Learning + Streamlit</p>", unsafe_allow_html=True)
