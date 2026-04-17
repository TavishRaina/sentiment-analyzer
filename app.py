import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page config
st.set_page_config(
    page_title="AI Sentiment Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# ---------- UI STYLE ----------
st.markdown("""
<style>

/* FORCE LIGHT MODE COMPLETELY */
html, body, .stApp {
    background-color: #f8fafc !important;
    color: #1e293b !important;
}

/* Main title centered */
h1 {
    text-align: center !important;
    color: #1e293b !important;
}

/* Other headings left aligned */
h2, h3 {
    text-align: left !important;
    color: #1e293b !important;
    margin-bottom: 10px;
}

/* Card blocks */
.block {
    padding: 20px;
    border-radius: 10px;
    background: white !important;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}

/* Text area */
textarea {
    background-color: white !important;
    color: #1e293b !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}

/* Input fields */
input {
    background-color: white !important;
    color: #1e293b !important;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #6366f1, #3b82f6);
    color: white !important;
    border-radius: 8px;
    border: none;
}

/* Fix ALL text visibility */
* {
    color: #1e293b !important;
}

/* Prevent dark overlay containers */
[data-testid="stAppViewContainer"] {
    background-color: #f8fafc !important;
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
            data = vectorizer.transform([user_input])
            prediction = model.predict(data)[0]
            prob = model.predict_proba(data)[0][prediction]

            st.session_state.history.append((user_input, prediction, prob))

            st.subheader("🔍 Result")

            if prediction == 1:
                st.success("✅ Positive Review")
            else:
                st.error("❌ Negative Review")

            st.progress(prob)
            st.write(f"Confidence Score: **{prob:.2f}**")

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
        st.write("Processing reviews...")

        predictions = model.predict(vectorizer.transform(df["review"]))
        df["Sentiment"] = ["Positive" if p == 1 else "Negative" for p in predictions]

        st.dataframe(df.head())

        pos = sum(predictions)
        neg = len(predictions) - pos

        st.success(f"Positive: {pos} | Negative: {neg}")
    else:
        st.error("CSV must contain 'review' column")

# ---------- BULK REVIEW ANALYZER + INSIGHTS ----------
st.divider()
st.subheader("📂 Bulk Review Analyzer")

multi_input = st.text_area("Paste multiple reviews (one per line):", height=200)

colA, colB = st.columns(2)

analyze_clicked = colA.button("Analyze Reviews")
insight_clicked = colB.button("Generate Insights")

if analyze_clicked or insight_clicked:
    if multi_input:
        reviews = multi_input.split("\n")
        reviews = [r.strip() for r in reviews if r.strip()]

        results = model.predict(vectorizer.transform(reviews))

        pos, neg = 0, 0

        pos_reviews = []
        neg_reviews = []

        st.subheader("🔍 Results")

        for r, p in zip(reviews, results):
            if p == 1:
                st.success(f"✅ {r}")
                pos += 1
                pos_reviews.append(r)
            else:
                st.error(f"❌ {r}")
                neg += 1
                neg_reviews.append(r)

            st.session_state.history.append((r, p, 1.0))

        st.subheader("📊 Summary")
        st.write(f"Positive: {pos}")
        st.write(f"Negative: {neg}")

        # ---------- INSIGHTS ----------
        if insight_clicked:
            st.divider()
            st.subheader("🧠 Smart Insights")

            pos_text = " ".join(pos_reviews).lower()
            neg_text = " ".join(neg_reviews).lower()

            liked = []
            disliked = []

            # 👍 POSITIVE PATTERNS
            if any(w in pos_text for w in ["good", "great", "amazing", "excellent", "love", "worth", "quality"]):
                liked.append("the overall quality and performance is appreciated")

            if any(w in pos_text for w in ["design", "look", "style", "color", "colour"]):
                liked.append("the design and appearance are liked")

            # 👎 NEGATIVE PATTERNS
            if any(w in neg_text for w in ["cost", "expensive", "price", "costly"]):
                disliked.append("many users feel the product is overpriced")

            if any(w in neg_text for w in ["bad", "poor", "worst", "terrible"]):
                disliked.append("there are concerns about poor performance")

            if any(w in neg_text for w in ["not good", "issue", "problem", "defect"]):
                disliked.append("users reported issues with certain features")

            if any(w in neg_text for w in ["color", "colour"]):
                disliked.append("some users are not satisfied with the product's appearance or color")

            # ---------- FINAL PARAGRAPH ----------
            if liked or disliked:
                final_text = "📌 Based on user reviews: "

                if liked:
                    final_text += "Users generally like that " + ", ".join(liked) + "."

                if disliked:
                    final_text += " However, " + ", ".join(disliked) + "."

                st.info(final_text)
            else:
                st.info("Not enough strong patterns detected to generate insights.")

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