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
def aspect_analysis(text):
    text = text.lower()

    aspects = ["camera", "battery", "performance", "display", "design", "looks", "size"]
    positive_words = ["good", "great", "amazing", "excellent", "decent phone", "outstanding", "perfect", "best", 
    "good phone"]
    negative_words = ["bad", "poor", "worst", "terrible", "fuck", "fucking", "fuckable",
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
    "trash"]

    parts = re.split(r'but|and|,', text)

    results = []

    for part in parts:
        for aspect in aspects:
            if aspect in part:
                if any(p in part for p in positive_words):
                    results.append((aspect, "Positive"))
                elif any(n in part for n in negative_words):
                    results.append((aspect, "Negative"))

    return results
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

               # 🔥 RULE OVERRIDES (FINAL ORDER)
                if any(phrase in cleaned for phrase in negative_phrases):
                    prediction = 0
                elif any(phrase in cleaned for phrase in positive_phrases):
                    prediction = 1
                elif any(word in cleaned for word in positive_keywords):
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

# ---------- ASPECT ANALYSIS ----------
st.subheader("🔎 Aspect Analysis")

aspects = aspect_analysis(user_input)

if aspects:
    for a, sentiment in aspects:
        if sentiment == "Positive":
            st.success(f"{a.capitalize()} → Positive")
        else:
            st.error(f"{a.capitalize()} → Negative")
else:
    st.info("No specific aspects detected")


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
                p = 0
            else:
                p = model.predict(vectorizer.transform([r]))[0]
                # 🔥 RULE OVERRIDES
                if any(phrase in r for phrase in negative_phrases):
                    p = 0
                elif any(phrase in r for phrase in positive_phrases):
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

colA, colB = st.columns(2)

analyze_clicked = colA.button("Analyze Reviews")
insight_clicked = colB.button("Generate Insights")

# ✅ ANALYZE BLOCK (only runs on Analyze)
if analyze_clicked:
    if multi_input:
        st.session_state.history = []   # 🔥 reset to stop duplication

        reviews = multi_input.split("\n")
        reviews = [r.strip() for r in reviews if r.strip()]

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

    else:
        st.warning("Please enter reviews")

# ✅ INSIGHTS BLOCK (separate, no duplication)
if insight_clicked:
    if multi_input:

        reviews = multi_input.split("\n")
        reviews = [r.strip() for r in reviews if r.strip()]

        cleaned_reviews = [clean_text(r) for r in reviews]

        pos_reviews = []
        neg_reviews = []

        for r in cleaned_reviews:
            if check_abuse(r):
                neg_reviews.append(r)
            else:
                p = model.predict(vectorizer.transform([r]))[0]

                if any(phrase in r for phrase in negative_phrases):
                    p = 0
                elif any(phrase in r for phrase in positive_phrases):
                    p = 1
                elif any(word in r for word in positive_keywords):
                    p = 1

                if p == 1:
                    pos_reviews.append(r)
                else:
                    neg_reviews.append(r)

        st.divider()
        st.subheader("🧠 Smart Insights")

        pos_text = " ".join(pos_reviews).lower()
        neg_text = " ".join(neg_reviews).lower()

        liked = []
        disliked = []

        if any(w in pos_text for w in ["good", "great", "amazing", "excellent", "love", "worth", "quality"]):
            liked.append("the overall quality and performance is appreciated")

        if any(w in pos_text for w in ["design", "look", "style", "color", "colour"]):
            liked.append("the design and appearance are liked")

        if any(w in neg_text for w in ["cost", "expensive", "price", "costly"]):
            disliked.append("many users feel the product is overpriced")

        if any(w in neg_text for w in ["bad", "poor", "worst", "terrible"]):
            disliked.append("there are concerns about poor performance")

        if any(w in neg_text for w in ["not good", "issue", "problem", "defect"]):
            disliked.append("users reported issues with certain features")

        if any(w in neg_text for w in ["color", "colour"]):
            disliked.append("some users are not satisfied with the product's appearance or color")

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
# ---------- HISTORY ----------
st.divider()
st.subheader("📜 Prediction History")

if st.session_state.history:
    for review, pred, prob in reversed(st.session_state.history):
        if pred == 1:
            st.success(f"✅ {review[:80]}... ({prob:.2f})")
        else:
            st.error(f"❌ {review[:80]}... ({prob:.2f})")
else:
    st.info("No predictions yet")

# ---------- FOOTER ----------
st.divider()
st.markdown("<p style='text-align:center;'>Built using Machine Learning + Streamlit</p>", unsafe_allow_html=True)
