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
