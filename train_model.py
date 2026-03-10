import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ================= LOAD DATA =================
df = pd.read_csv("text.csv")   # columns: text, label

# ================= PREPROCESS =================
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join(w for w in text.split() if w not in stop_words)
    text = ' '.join(stemmer.stem(w) for w in text.split())
    return text

df['text'] = df['text'].astype(str).apply(clean_text)

# ================= LABEL ENCODING =================
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['label'])

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label_enc'], test_size=0.2, random_state=42
)

# ================= VECTORIZER =================
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ================= MODEL =================
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ================= CHECK =================
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ================= SAVE FILES =================
joblib.dump(model, "emotion_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(le, "label_encoder.pkl")

print("✅ All .pkl files saved successfully")