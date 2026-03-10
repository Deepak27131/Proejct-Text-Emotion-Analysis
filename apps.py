# import streamlit as st
# import pandas as pd
# import re
# import joblib
# import matplotlib.pyplot as plt
# from collections import Counter
# from dotenv import load_dotenv
# import os
# from groq import Groq


# # ========== LOAD ==========
# model = joblib.load("emotion_model.pkl")
# vectorizer = joblib.load("vectorizer.pkl")

# label_map = {
#     0: "sadness",
#     1: "joy",
#     2: "anger",
#     3: "fear",
#     4: "unknown"
# }

# load_dotenv()
# client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# # ========== CLEAN ==========
# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r"[^\w\s]", "", text)
#     text = re.sub(r"\d+", "", text)
#     return text.strip()

# # ========== UI ==========
# st.set_page_config(layout="wide")
# st.title("📊 Emotion Analysis Dashboard + 💬 Chat")

# uploaded_file = st.file_uploader("Upload CSV (must contain 'text' column)", type=["csv"])

# # ================= CSV ANALYSIS =================
# if uploaded_file:
#     df = pd.read_csv(uploaded_file)

#     if "text" not in df.columns:
#         st.error("CSV must contain a 'text' column")
#         st.stop()

#     df["clean_text"] = df["text"].astype(str).apply(clean_text)
#     X_vec = vectorizer.transform(df["clean_text"])

#     preds = model.predict(X_vec)
#     df["emotion"] = [label_map.get(p, "unknown") for p in preds]

#     st.subheader("📄 Text-wise Emotion Analysis")
#     st.dataframe(df[["text", "emotion"]], use_container_width=True)

#     # ===== Emotion Count =====
#     emotion_counts = Counter(df["emotion"])

#     col1, col2 = st.columns(2)

#     # BAR CHART
#     with col1:
#         st.subheader("📊 Emotion Distribution (Bar)")
#         fig, ax = plt.subplots()
#         ax.bar(emotion_counts.keys(), emotion_counts.values())
#         plt.xticks(rotation=45)
#         st.pyplot(fig)

#     # PIE CHART
#     with col2:
#         st.subheader("🥧 Emotion Distribution (Pie)")
#         fig2, ax2 = plt.subplots()
#         ax2.pie(
#             emotion_counts.values(),
#             labels=emotion_counts.keys(),
#             autopct="%1.1f%%",
#             startangle=90
#         )
#         ax2.axis("equal")
#         st.pyplot(fig2)

# # ================= GROQ CHAT MODE =================
# st.markdown("---")
# st.subheader("💬 Groq Chat Mode (Ask about analysis or anything)")

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# user_msg = st.text_input("Ask something (about emotions, text, or anything)")

# if st.button("Send"):
#     if user_msg.strip():
#         st.session_state.chat_history.append(("user", user_msg))

#         response = client.chat.completions.create(
#             model="llama-3.3-70b-versatile",
#             messages=[
#                 {"role": "system", "content": "You are an emotion analysis assistant."},
#                 *[
#                     {"role": role, "content": msg}
#                     for role, msg in st.session_state.chat_history
#                 ]
#             ]
#         )

#         bot_reply = response.choices[0].message.content
#         st.session_state.chat_history.append(("assistant", bot_reply))

# # SHOW CHAT
# for role, msg in st.session_state.chat_history:
#     if role == "user":
#         st.markdown(f"**🧑 You:** {msg}")
#     else:
#         st.markdown(f"**🤖 Groq:** {msg}")



import streamlit as st
import pandas as pd
import re
import joblib
import matplotlib.pyplot as plt
from collections import Counter
from dotenv import load_dotenv
import os
import google.generativeai as genai

# ========== LOAD ENV ==========
load_dotenv()

# ========== LOAD MODEL ==========
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

label_map = {
    0: "sadness",
    1: "joy",
    2: "anger",
    3: "fear",
    4: "unknown"
}

# ========== GEMINI CONFIG ==========
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# ========== CLEAN TEXT ==========
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    return text.strip()

# ========== EMOTION PREDICT ==========
def predict_emotion(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]

    if hasattr(model, "predict_proba"):
        confidence = max(model.predict_proba(vec)[0])
    else:
        confidence = 1.0

    return label_map.get(pred, "unknown"), confidence

# ========== UI ==========
st.set_page_config(layout="wide")
st.title("📊 Emotion Analysis Dashboard + 💬 Gemini Chat")

uploaded_file = st.file_uploader(
    "Upload CSV (must contain 'text' column)", type=["csv"]
)

# ================= CSV ANALYSIS =================
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "text" not in df.columns:
        st.error("CSV must contain a 'text' column")
        st.stop()

    df["clean_text"] = df["text"].astype(str).apply(clean_text)
    X_vec = vectorizer.transform(df["clean_text"])

    preds = model.predict(X_vec)
    df["emotion"] = [label_map.get(p, "unknown") for p in preds]

    st.subheader("📄 Text-wise Emotion Analysis")
    st.dataframe(df[["text", "emotion"]], use_container_width=True)

    emotion_counts = Counter(df["emotion"])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Emotion Distribution (Bar)")
        fig, ax = plt.subplots()
        ax.bar(emotion_counts.keys(), emotion_counts.values())
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col2:
        st.subheader("🥧 Emotion Distribution (Pie)")
        fig2, ax2 = plt.subplots()
        ax2.pie(
            emotion_counts.values(),
            labels=emotion_counts.keys(),
            autopct="%1.1f%%",
            startangle=90
        )
        ax2.axis("equal")
        st.pyplot(fig2)

# ================= CHAT MODE =================
st.markdown("---")
st.subheader("💬 Gemini Chat Mode")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_msg = st.text_input("Ask something (about emotions, text, or anything)")

if st.button("Send"):
    if user_msg.strip():
        # 1️⃣ Emotion FIRST
        emotion, confidence = predict_emotion(user_msg)
        st.session_state.chat_history.insert(
            0, ("emotion", f"{emotion} (confidence: {round(confidence, 2)})")
        )

        # 2️⃣ User message
        st.session_state.chat_history.insert(0, ("user", user_msg))

        # 3️⃣ Build prompt for Gemini
        prompt = "You are an emotion analysis assistant.\n"
        for role, msg in reversed(st.session_state.chat_history):
            if role == "user":
                prompt += f"User: {msg}\n"
            elif role == "assistant":
                prompt += f"Assistant: {msg}\n"

        # 4️⃣ Gemini response
        response = gemini_model.generate_content(prompt)
        bot_reply = response.text

        # 5️⃣ Store reply (TOP)
        st.session_state.chat_history.insert(0, ("assistant", bot_reply))

# ================= SHOW CHAT =================
for role, msg in st.session_state.chat_history:
    if role == "emotion":
        st.markdown(f"🧠 **Detected Emotion:** `{msg}`")
    elif role == "user":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Gemini:** {msg}")