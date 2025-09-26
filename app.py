import streamlit as st
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# --------------------------
# Simple Offline Sentiment
# --------------------------
POSITIVE_WORDS = {"good", "great", "happy", "excellent", "love", "nice", "awesome"}
NEGATIVE_WORDS = {"bad", "sad", "terrible", "hate", "poor", "worst", "awful"}

def offline_sentiment(text: str):
    text = text.lower().split()
    pos = sum(1 for w in text if w in POSITIVE_WORDS)
    neg = sum(1 for w in text if w in NEGATIVE_WORDS)
    total = len(text) if len(text) > 0 else 1
    return [(pos - neg)/total, neg/total]

# --------------------------
# Phase Functions
# --------------------------
def lexical_phase(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = [w for w in text.split() if len(w) > 1]
    return " ".join(tokens)

def syntactic_phase(text: str) -> str:
    return " ".join(["WORD" for w in text.split()])  # simple placeholder

def semantic_phase(text: str):
    return offline_sentiment(text)

def discourse_phase(text: str) -> str:
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    first_words = [s.split()[0] for s in sentences if s.split()]
    return f"{len(sentences)} {' '.join(first_words)}"

PRAG_TOKENS = ["must", "should", "might", "could", "will", "?", "!"]
def pragmatic_phase(text: str):
    lower = text.lower()
    return [lower.count(t) for t in PRAG_TOKENS]

# --------------------------
# Model Evaluation
# --------------------------
def compare_models(X, y):
    models = {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=300),
        "SVM": SVC()
    }
    results = {}
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    for name, model in models.items():
        try:
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            results[name] = round(accuracy_score(yte, pred) * 100, 2)
        except Exception as e:
            results[name] = f"Error: {e}"
    return results

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title=" NLP Phase-wise Analysis", layout="wide", page_icon="🧠")
st.markdown(
    "<h1 style='text-align: center; color: #4B0082;'>🧠 NLP Phase-wise Model Comparison</h1>"
    "<p style='text-align: center; color: gray;'>Works fully offline. Upload CSV, choose phase, benchmark ML models.</p>",
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("📂 Upload & Settings")
    csv_file = st.file_uploader("Upload CSV file", type="csv")
    st.markdown("---")
    st.markdown("💡 CSV should have text and label columns. Select phase and run comparison.")

if not csv_file:
    st.info("⬅️ Upload a dataset to start.")
    st.stop()

data = pd.read_csv(csv_file)
st.subheader("📊 Data Preview")
st.dataframe(data.head(), use_container_width=True)

c1, c2 = st.columns(2)
text_col = c1.selectbox("Text Column", data.columns)
label_col = c2.selectbox("Target/Label Column", data.columns)

phase = st.selectbox(
    "Select NLP Phase",
    ["Lexical", "Syntactic", "Semantic", "Discourse", "Pragmatic"]
)

if st.button("🚀 Run Comparison"):
    X_text = data[text_col].astype(str)
    y = data[label_col]

    # Feature engineering
    if phase == "Lexical":
        feats = X_text.apply(lexical_phase)
        X = CountVectorizer().fit_transform(feats)
    elif phase == "Syntactic":
        feats = X_text.apply(syntactic_phase)
        X = CountVectorizer().fit_transform(feats)
    elif phase == "Semantic":
        X = pd.DataFrame(X_text.apply(semantic_phase).tolist(), columns=["Sentiment", "NegProportion"])
    elif phase == "Discourse":
        feats = X_text.apply(discourse_phase)
        X = CountVectorizer().fit_transform(feats)
    else:  # Pragmatic
        X = pd.DataFrame(X_text.apply(pragmatic_phase).tolist(), columns=PRAG_TOKENS)

    results = compare_models(X, y)

    # --------------------------
    # Filter numeric results safely
    # --------------------------
    res_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
    res_df = res_df[res_df["Accuracy"].apply(lambda x: isinstance(x, (int, float)))]
    res_df = res_df.sort_values("Accuracy", ascending=False)

    st.subheader("🏆 Model Accuracy")
    st.table(res_df)

    # --------------------------
    # Plot results
    # --------------------------
    fig, ax = plt.subplots(figsize=(8,5))
    bars = ax.bar(res_df["Model"], res_df["Accuracy"], color=["#4B0082","#6A5ACD","#9370DB","#BA55D3"], alpha=0.8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0,100)
    ax.set_title(f"Performance on {phase} Features", fontsize=14, fontweight="bold")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval+1, f"{yval:.1f}%", ha='center', fontsize=10)
    st.pyplot(fig)
