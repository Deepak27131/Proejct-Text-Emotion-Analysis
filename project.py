# Save this as app.py
import streamlit as st

# Title
st.title("AI-Powered Text Emotion Analyzer")

# Upload CSV file
st.subheader("Upload CSV File (Optional)")
uploaded_file = st.file_uploader("Upload CSV with 'text' column", type=["csv"])

# Input text box
st.subheader("Enter Text")
user_input = st.text_area("Type your text here:", height=100)

# Button to analyze
if st.button("Analyze Emotion"):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        texts = df['text'].tolist()
    elif user_input:
        texts = [user_input]
    else:
        st.error("Please enter text or upload a CSV file.")
        st.stop()

    # Preprocess and predict
    results = []
    for text in texts:
        # Preprocess
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = ' '.join([word for word in text.split() if word not in stop_words])
        text = ' '.join([stemmer.stem(word) for word in text.split()])
        
        # Vectorize
        text_vec = vectorizer.transform([text])
        
        # Predict
        emotion = model.predict(text_vec)
        confidence = model.predict_proba(text_vec).max()
        
        # Suggestion
        suggestions = {
            "sadness": "Try to rephrase: 'I feel a bit down today' instead of 'I am so sad'.",
            "anger": "Try to rephrase: 'I am frustrated' instead of 'I hate this'.",
            "love": "Keep it! It's a positive emotion.",
            "surprise": "Try to rephrase: 'I am shocked' instead of 'I am so surprised'.",
            "fear": "Try to rephrase: 'I am anxious' instead of 'I am scared'.",
            "joy": "Keep it! It's a positive emotion."
        }
        suggestion = suggestions.get(emotion, "No suggestion available.")

        results.append({
            "text": text,
            "emotion": emotion,
            "confidence": f"{confidence * 100:.2f}%",
            "suggestion": suggestion
        })

    # Display results
    for result in results:
        st.write(f"**Text:** {result['text']}")
        st.write(f"**Emotion:** {result['emotion'].title()}")
        st.write(f"**Confidence:** {result['confidence']}")
        st.write(f"**Suggestion:** {result['suggestion']}")
        st.markdown("---")