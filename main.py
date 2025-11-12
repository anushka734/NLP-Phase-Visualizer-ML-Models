import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import csv
from urllib.parse import urljoin
import io
import time
import random
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# imbalanced tools
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# NLP & ML
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

# -------------------------
# Configuration / constants
# -------------------------
DATA_FILE = "politifact_data.csv"
FOLDS = 5

# -------------------------
# Robust spaCy loader
# -------------------------
@st.cache_resource
def init_spacy():
    """
    Try to load an installed spaCy model. If missing, print guidance
    so users can add the model wheel into requirements.txt (for deployment).
    """
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        st.error("spaCy model 'en_core_web_sm' not available. Add the model wheel line to requirements.txt.")
        st.code("""
# Example: add this to requirements.txt for Streamlit Cloud
https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
imbalanced-learn
lxml
        """, language="text") # <--- ADDED lxml HERE
        raise

# initialize nlp or stop early
try:
    NLP = init_spacy()
except Exception:
    st.stop()

STOPWORDS = STOP_WORDS
PRAG_WORDS = ["must", "should", "might", "could", "will", "?", "!"]

# -------------------------
# 1) Scraper: fetch Politifact claims by date range (OPTIMIZED)
# -------------------------
def fetch_politifact_claims(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """
    Crawl Politifact list pages using requests.Session and lxml for speed.
    Stops when claims older than start_ts appear or pages end.
    Returns a cleaned DataFrame saved to DATA_FILE.
    """
    base_url = "https://www.politifact.com/factchecks/list/"
    current_url = base_url
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["author", "statement", "source", "date", "label"])
    
    scraped_rows_count = 0
    page_count = 0
    
    status_slot = st.empty()
    status_slot.caption(f"Starting scrape from {start_ts.strftime('%Y-%m-%d')} to {end_ts.strftime('%Y-%m-%d')}")

    # <--- OPTIMIZATION 1: Initialize a session object to reuse connections
    session = requests.Session()

    while current_url and page_count < 100: 
        page_count += 1
        status_slot.text(f"Fetching page {page_count}... Scraped {scraped_rows_count} claims so far.")

        try:
            # <--- OPTIMIZATION 2: Use session.get() instead of requests.get()
            response = session.get(current_url, timeout=15)
            response.raise_for_status()
            
            # <--- OPTIMIZATION 3: Use 'lxml' for much faster HTML parsing
            soup = BeautifulSoup(response.text, "lxml") 
        
        except requests.exceptions.RequestException as e:
            status_slot.error(f"Network Error during request: {e}. Stopping scrape.")
            break

        rows_to_add = []

        for card in soup.find_all("li", class_="o-listicle__item"):
            date_div = card.find("div", class_="m-statement__desc")
            date_text = date_div.get_text(strip=True) if date_div else None
            claim_date = None
            
            if date_text:
                match = re.search(r"stated on ([A-Za-z]+\s+\d{1,2},\s+\d{4})", date_text)
                if match:
                    try:
                        claim_date = pd.to_datetime(match.group(1), format='%B %d, %Y')
                    except ValueError:
                        continue
            
            if claim_date:
                # Check date range
                if start_ts <= claim_date <= end_ts:
                    statement_block = card.find("div", class_="m-statement__quote")
                    statement = statement_block.find("a", href=True).get_text(strip=True) if statement_block and statement_block.find("a", href=True) else None
                    source_a = card.find("a", class_="m-statement__name")
                    source = source_a.get_text(strip=True) if source_a else None
                    footer = card.find("footer", class_="m-statement__footer")
                    author = None
                    if footer:
                        author_match = re.search(r"By\s+([^•]+)", footer.get_text(strip=True))
                        if author_match:
                            author = author_match.group(1).strip()
                            
                    label_img = card.find("img", alt=True)
                    label = label_img['alt'].replace('-', ' ').title() if label_img and 'alt' in label_img.attrs else None

                    rows_to_add.append([author, statement, source, claim_date.strftime('%Y-%m-%d'), label])

                # This "stop early" logic is correct and saves time
                elif claim_date < start_ts:
                    status_slot.warning(f"Encountered claim older than start date ({start_ts.strftime('%Y-%m-%d')}). Stopping scrape.")
                    current_url = None
                    break 

        if current_url is None:
            break

        writer.writerows(rows_to_add)
        scraped_rows_count += len(rows_to_add)

        next_link = soup.find("a", class_="c-button c-button--hollow", string=re.compile(r"Next", re.I))
        if next_link and 'href' in next_link.attrs:
            next_href = next_link['href'].rstrip('&').rstrip('?')
            current_url = urljoin(base_url, next_href)
        else:
            status_slot.success("No more pages found or last page reached.")
            current_url = None

    status_slot.success(f"Scraping finished! Total claims processed: {scraped_rows_count}")
    
    output.seek(0)
    df = pd.read_csv(output, header=0, keep_default_na=False)
    df = df.dropna(subset=['statement', 'label'])
    
    # Use the DATA_FILE constant defined at the top
    df.to_csv(DATA_FILE, index=False) 
    return df

# -------------------------
# 2) Feature extraction helpers (lexical / syntactic / semantic / discourse / pragmatic)
# -------------------------
def extract_lexical(text: str) -> str:
    doc = NLP(text.lower())
    lemmas = [tok.lemma_ for tok in doc if tok.is_alpha and tok.text not in STOPWORDS]
    return " ".join(lemmas)

def extract_syntactic(text: str) -> str:
    doc = NLP(text)
    tags = " ".join([tok.pos_ for tok in doc])
    return tags

def extract_semantic(text: str):
    blob = TextBlob(text)
    return [blob.sentiment.polarity, blob.sentiment.subjectivity]

def extract_discourse(text: str) -> str:
    doc = NLP(text)
    sents = [s.text.strip() for s in doc.sents if s.text.strip()]
    # store the number of sentences and the first token of each sentence to hint structure
    first_tokens = " ".join([s.split()[0].lower() for s in sents if len(s.split()) > 0])
    return f"{len(sents)} {first_tokens}"

def extract_pragmatic(text: str):
    t = text.lower()
    return [t.count(w) for w in PRAG_WORDS]

# -------------------------
# 3) Model helpers & feature transformation
# -------------------------
def create_classifier(kind: str):
    if kind == "Naive Bayes":
        return MultinomialNB()
    if kind == "Decision Tree":
        return DecisionTreeClassifier(random_state=42, class_weight="balanced")
    if kind == "Logistic Regression":
        return LogisticRegression(max_iter=1000, solver="liblinear", random_state=42, class_weight="balanced")
    if kind == "SVM":
        return SVC(kernel="linear", C=0.5, random_state=42, class_weight="balanced")
    return None

def transform_features(raw_series: pd.Series, phase: str, existing_vectorizer=None):
    """
    Convert raw statements into model-ready features depending on 'phase'.
    Returns (features_matrix_or_df, fitted_vectorizer_or_None)
    """
    if phase == "Lexical & Morphological":
        processed = raw_series.apply(extract_lexical)
        vec = existing_vectorizer if existing_vectorizer else CountVectorizer(binary=True, ngram_range=(1,2))
        X = vec.fit_transform(processed)
        return X, vec

    if phase == "Syntactic":
        processed = raw_series.apply(extract_syntactic)
        vec = existing_vectorizer if existing_vectorizer else TfidfVectorizer(max_features=5000)
        X = vec.fit_transform(processed)
        return X, vec

    if phase == "Semantic":
        arr = raw_series.apply(extract_semantic).tolist()
        df = pd.DataFrame(arr, columns=["polarity", "subjectivity"])
        return df, None

    if phase == "Discourse":
        processed = raw_series.apply(extract_discourse)
        vec = existing_vectorizer if existing_vectorizer else CountVectorizer(ngram_range=(1,2), max_features=5000)
        X = vec.fit_transform(processed)
        return X, vec

    if phase == "Pragmatic":
        arr = raw_series.apply(extract_pragmatic).tolist()
        df = pd.DataFrame(arr, columns=PRAG_WORDS)
        return df, None

    return None, None

# -------------------------
# 4) Training & cross-validated evaluation (SMOTE + Stratified K-Fold)
# -------------------------
def run_benchmark(df: pd.DataFrame, chosen_phase: str) -> pd.DataFrame:
    """
    Map labels to binary target, extract features for chosen_phase, run Stratified K-Fold with SMOTE
    and return an aggregated metrics DataFrame.
    """
    REAL = ["True", "No Flip", "Mostly True", "Half Flip", "Half True"]
    FAKE = ["False", "Barely True", "Pants On Fire", "Full Flop"]

    def to_binary(lbl):
        if lbl in REAL:
            return 1
        if lbl in FAKE:
            return 0
        return np.nan

    df = df.copy()
    df["target_label"] = df["label"].apply(to_binary)
    df = df.dropna(subset=["target_label"])
    df = df[df["statement"].astype(str).str.len() > 10]

    X_raw = df["statement"].astype(str)
    y = df["target_label"].astype(int).values

    if len(np.unique(y)) < 2:
        st.error("Binary mapping produced fewer than 2 classes. Adjust dataset or labels.")
        return pd.DataFrame()

    X_full, vectorizer = transform_features(X_raw, chosen_phase)
    if X_full is None:
        st.error("Feature extraction returned nothing — check phase choice.")
        return pd.DataFrame()

    # convert DataFrame features to numpy if needed
    if isinstance(X_full, pd.DataFrame):
        X_full = X_full.values

    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

    models = {
        "Naive Bayes": create_classifier("Naive Bayes"),
        "Decision Tree": create_classifier("Decision Tree"),
        "Logistic Regression": create_classifier("Logistic Regression"),
        "SVM": create_classifier("SVM"),
    }

    results = {}

    raw_list = X_raw.tolist()

    for name, model in models.items():
        st.caption(f"Training {name} — {FOLDS}-fold CV with SMOTE where appropriate...")
        metrics_accum = {
            "accuracy": [], "f1": [], "precision": [], "recall": [], "train_time": [], "inference_time": []
        }

        for fold, (tr_idx, te_idx) in enumerate(skf.split(X_full, y)):
            X_tr_raw = pd.Series([raw_list[i] for i in tr_idx])
            X_te_raw = pd.Series([raw_list[i] for i in te_idx])
            y_tr = y[tr_idx]
            y_te = y[te_idx]

            # vectorize / transform for this fold
            if vectorizer is not None:
                # lexical phase requires the lexical function; others may require syntactic pipeline
                if "Lexical" in chosen_phase:
                    X_tr = vectorizer.transform(X_tr_raw.apply(extract_lexical))
                    X_te = vectorizer.transform(X_te_raw.apply(extract_lexical))
                else:
                    X_tr = vectorizer.transform(X_tr_raw.apply(extract_syntactic if "Syntactic" in chosen_phase else extract_lexical))
                    X_te = vectorizer.transform(X_te_raw.apply(extract_syntactic if "Syntactic" in chosen_phase else extract_lexical))
            else:
                X_tr_df, _ = transform_features(X_tr_raw, chosen_phase)
                X_te_df, _ = transform_features(X_te_raw, chosen_phase)
                X_tr = X_tr_df
                X_te = X_te_df

            start_train = time.time()
            try:
                if name == "Naive Bayes":
                    X_tr_fixed = np.abs(X_tr).astype(float)
                    clf = model
                    clf.fit(X_tr_fixed, y_tr)
                else:
                    # SMOTE pipeline for other models
                    pipe = ImbPipeline([("sampler", SMOTE(random_state=42, k_neighbors=3)), ("clf", model)])
                    pipe.fit(X_tr, y_tr)
                    clf = pipe
                train_time = time.time() - start_train

                start_inf = time.time()
                y_pred = clf.predict(X_te)
                inf_time = (time.time() - start_inf) * 1000.0  # ms

                metrics_accum["accuracy"].append(accuracy_score(y_te, y_pred))
                metrics_accum["f1"].append(f1_score(y_te, y_pred, average="weighted", zero_division=0))
                metrics_accum["precision"].append(precision_score(y_te, y_pred, average="weighted", zero_division=0))
                metrics_accum["recall"].append(recall_score(y_te, y_pred, average="weighted", zero_division=0))
                metrics_accum["train_time"].append(train_time)
                metrics_accum["inference_time"].append(inf_time)

            except Exception as exc:
                st.warning(f"Fold {fold+1} for {name} failed: {exc}")
                # append zeros to keep array lengths consistent
                for k in metrics_accum:
                    metrics_accum[k].append(0)

        # aggregate
        if metrics_accum["accuracy"]:
            results[name] = {
                "Model": name,
                "Accuracy": np.mean(metrics_accum["accuracy"]) * 100.0,
                "F1-Score": np.mean(metrics_accum["f1"]),
                "Precision": np.mean(metrics_accum["precision"]),
                "Recall": np.mean(metrics_accum["recall"]),
                "Training Time (s)": round(np.mean(metrics_accum["train_time"]), 2),
                "Inference Latency (ms)": round(np.mean(metrics_accum["inference_time"]), 2),
            }
        else:
            st.error(f"{name} failed for all folds.")
            results[name] = {
                "Model": name, "Accuracy": 0, "F1-Score": 0, "Precision": 0, "Recall": 0,
                "Training Time (s)": 0, "Inference Latency (ms)": 9999
            }

    return pd.DataFrame(list(results.values()))

# -------------------------
# 5) Humor / critique utilities
# -------------------------
def phase_roast(phase: str) -> str:
    r = {
        "Lexical & Morphological": [
            "Lexical stage: when word counts moonlight as insight.",
            "Counting words like a vintage librarian with a data fetish."
        ],
        "Syntactic": [
            "Syntactic wins: grammar police in the house.",
            "This one judged the sentence structure and won the talent show."
        ],
        "Semantic": [
            "Semantic: feeling the vibes and calling it science.",
            "It judged tone and mood, and apparently that's enough."
        ],
        "Discourse": [
            "Discourse: the argument auditor. It read the essay and filed a complaint.",
            "It cared about flow and beat the rest with rhetorical elegance."
        ],
        "Pragmatic": [
            "Pragmatic got it right by focusing on intention — the Sherlock of NLP.",
            "It sniffed out the words that betray intent and did the detective work."
        ]
    }
    return random.choice(r.get(phase, ["The results are inconclusive; the models are napping."]))

def model_roast(model: str) -> str:
    r = {
        "Naive Bayes": [
            "Naive Bayes: quick, naïve, and occasionally brilliant.",
            "The fast-food champion — cheap, reliable, surprisingly filling."
        ],
        "Decision Tree": [
            "Decision Tree: it split until the truth surrendered.",
            "A very decisive model — it just says 'yes' or 'no' until things make sense."
        ],
        "Logistic Regression": [
            "Logistic Regression: the straight-arrow, inevitable winner.",
            "It draws a line and says 'this is all I need'."
        ],
        "SVM": [
            "SVM: built a moat and called it a classifier.",
            "Margin maximizer and proud of it."
        ]
    }
    return random.choice(r.get(model, ["The AI has no comment."]))

def compose_roast(results_df: pd.DataFrame, phase: str) -> str:
    if results_df.empty:
        return "No models were trained. The ML interns are on strike."
    results_df["F1-Score"] = pd.to_numeric(results_df["F1-Score"], errors="coerce").fillna(0)
    best = results_df.loc[results_df["F1-Score"].idxmax()]
    best_name = best["Model"]
    best_f1 = best["F1-Score"]
    best_acc = best["Accuracy"]

    header = f"🏆 Winner: {best_name} — {best_acc:.2f}% accuracy, F1 {best_f1:.2f}"
    body = (
        f"{phase_roast(phase)}\n\n"
        f"{model_roast(best_name)}\n\n"
        f"*Note: models are particularly perplexed by 'Mostly True' — a label that confuses humans and machines alike.*"
    )
    return header + "\n\n" + body

# -------------------------
# 6) Cosine similarity helper for UI
# -------------------------
def compute_and_display_similarity(a: str, b: str):
    if not a or not b:
        st.info("Enter two texts to compare their cosine similarity (TF-IDF).")
        return
    vect = TfidfVectorizer()
    vecs = vect.fit_transform([a, b])
    score = float(cosine_similarity(vecs[0], vecs[1])[0][0])
    st.metric("Cosine similarity", f"{score*100:.2f}%")
    st.progress(min(max(score, 0.0), 1.0))

# -------------------------
# 7) Streamlit application (dark glowing UI + glowing plot)
# -------------------------
def run_app():
    st.set_page_config(page_title="Truth Lens — Dark Neon", layout="wide", page_icon="🛰️")

    # ---------- Dark neon CSS ----------
    st.markdown(
        """
        <style>
        :root{
            --bg:#0b0f14;
            --card:#0f1720;
            --muted:#9aa4b2;
            --neon:#00ffd1;
            --neon-2:#6ee7ff;
            --glass: rgba(255,255,255,0.04);
        }
        /* Page background and text */
        body, .css-1d391kg { background: var(--bg); color: #e6eef6; }
        .stApp { background: linear-gradient(180deg, rgba(5,7,10,1) 0%, rgba(10,12,16,1) 100%); }

        /* Neon header */
        .neon-header{
            text-align:center;
            padding:22px;
            border-radius:14px;
            color: var(--neon);
            background: linear-gradient(90deg, rgba(0,255,209,0.03), rgba(110,231,255,0.02));
            box-shadow: 0 0 18px rgba(0,255,209,0.06), inset 0 -1px 0 rgba(255,255,255,0.02);
            backdrop-filter: blur(4px);
        }
        .neon-title { font-size:34px; font-weight:700; text-shadow: 0 0 12px rgba(0,255,209,0.25); margin:0; }
        .neon-sub { color: var(--neon-2); margin-top:6px; font-size:13px; opacity:0.9; }

        /* Cards */
        .glass-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
            border: 1px solid rgba(110,231,255,0.06);
            border-radius:12px;
            padding:18px;
            box-shadow: 0 6px 30px rgba(2,6,23,0.6);
            transition: transform .25s ease, box-shadow .25s ease;
        }
        .glass-card:hover { transform: translateY(-6px); box-shadow: 0 18px 60px rgba(0,255,209,0.06); }

        /* Buttons */
        div.stButton > button {
            background: linear-gradient(90deg, #00ffd1, #6ee7ff);
            color: #022626;
            border-radius:10px;
            padding:0.6rem 1rem;
            font-weight:700;
            box-shadow: 0 6px 20px rgba(0,255,209,0.08);
            border: none;
        }
        div.stButton > button:hover {
            transform: scale(1.03);
            box-shadow: 0 12px 36px rgba(0,255,209,0.12);
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(8,10,12,0.95), rgba(14,18,20,0.95));
            border-right: 1px solid rgba(110,231,255,0.04);
        }
        .sidebar .stButton > button { width:100%; }

        /* small muted text */
        .muted { color: var(--muted); font-size:13px; }

        /* footer */
        .app-footer { text-align:center; color: #90a4b2; margin-top:20px; font-size:13px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Header ----------
    st.markdown(
        """
        <div class="neon-header">
            <div class="neon-title">🛰️ The AI Deception Detector </div>
            <div class="neon-sub">Scrape • Featurize • Validate • Roast — AI fact check</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")  # spacing
    st.divider()

    # Columns layout
    left_col, center_col, right_col = st.columns([1.1, 2.0, 1.2])

    # ---------- LEFT: Data & Controls ----------
    with left_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🔎 Data Source")
        st.markdown('<div class="muted">Harvest fact-check statements from the Politifact archive.</div>', unsafe_allow_html=True)

        earliest = pd.to_datetime("2007-01-01")
        latest = pd.to_datetime("today").normalize()

        start_date = st.date_input("Start date", min_value=earliest, max_value=latest, value=pd.to_datetime("2023-01-01"))
        end_date = st.date_input("End date", min_value=earliest, max_value=latest, value=latest)

        if st.button("⛏️ Fetch claims"):
            if pd.to_datetime(start_date) > pd.to_datetime(end_date):
                st.error("Start must not be after End.")
            else:
                with st.spinner("Crawling Politifact — this might take a few minutes..."):
                    # This now calls your optimized function
                    df = fetch_politifact_claims(pd.to_datetime(start_date), pd.to_datetime(end_date)) 
                if not df.empty:
                    st.success(f"Scrape done — {len(df)} statements saved.")
                    st.session_state["scraped_df"] = df
                else:
                    st.warning("No results. Try a wider date range.")

        st.markdown("---")
        st.markdown("### ⚙️ Analysis Settings")
        phases = ["Lexical & Morphological", "Syntactic", "Semantic", "Discourse", "Pragmatic"]
        chosen_phase = st.selectbox("Feature group", phases, index=0)

        if st.button("🚀 Run evaluation"):
            if "scraped_df" not in st.session_state or st.session_state["scraped_df"].empty:
                st.error("Please fetch data first.")
            else:
                with st.spinner(f"Running {FOLDS}-Fold CV with SMOTE on {chosen_phase} features..."):
                    df_res = run_benchmark(st.session_state["scraped_df"], chosen_phase)
                    st.session_state["df_results"] = df_res
                    st.session_state["phase_used"] = chosen_phase
                st.success("Benchmark complete.")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- CENTER: Results & Visuals ----------
    with center_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Benchmark Results")

        if "df_results" not in st.session_state or st.session_state["df_results"].empty:
            st.info("Run the evaluation from the left panel to see results here.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            df_res = st.session_state["df_results"]
            phase_run = st.session_state.get("phase_used", "N/A")

            # Summary cards row (small neon stat cards)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                best_acc = df_res["Accuracy"].max() if not df_res.empty else 0
                st.metric(label="Best Accuracy", value=f"{best_acc:.2f}%")
            with c2:
                best_f1 = df_res["F1-Score"].max() if not df_res.empty else 0
                st.metric(label="Best F1-Score", value=f"{best_f1:.2f}")
            with c3:
                avg_train = df_res["Training Time (s)"].mean() if not df_res.empty else 0
                st.metric(label="Avg Train (s)", value=f"{avg_train:.2f}")
            with c4:
                avg_inf = df_res["Inference Latency (ms)"].mean() if not df_res.empty else 0
                st.metric(label="Avg Inference (ms)", value=f"{avg_inf:.2f}")

            st.markdown(f"**Feature set used:** {phase_run}")
            st.dataframe(df_res[["Model", "Accuracy", "F1-Score", "Training Time (s)", "Inference Latency (ms)"]], height=220, use_container_width=True)

            st.markdown("---")
            st.markdown("### 🔬 Metric comparison")
            metric_options = ["Accuracy", "F1-Score", "Precision", "Recall", "Training Time (s)", "Inference Latency (ms)"]
            chosen_metric = st.selectbox("Choose metric to plot", metric_options, index=1, key="metric_plot")

            df_plot = df_res[["Model", chosen_metric]].set_index("Model")
            st.bar_chart(df_plot)

            st.markdown("---")
            st.markdown("### 🎭 Model Roast")
            roast_text = compose_roast(df_res, phase_run)
            st.markdown(roast_text)
            st.markdown("</div>", unsafe_allow_html=True)

    # ---------- RIGHT: Tools & Cosine Similarity ----------
    with right_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🛠️ Quick Tools")
        st.markdown('<div class="muted">Inspect models, compare texts, and visualize speed vs quality.</div>', unsafe_allow_html=True)

        if "df_results" in st.session_state and not st.session_state["df_results"].empty:
            st.markdown("---")
            st.markdown("#### ⚖️ Speed vs Quality")
            m_quality = ["Accuracy", "F1-Score", "Precision", "Recall"]
            m_speed = ["Training Time (s)", "Inference Latency (ms)"]
            x = st.selectbox("X (speed)", m_speed, index=1, key="speed_x")
            y = st.selectbox("Y (quality)", m_quality, index=0, key="quality_y")

            fig, ax = plt.subplots(figsize=(5, 3))
            df_r = st.session_state["df_results"]
            ax.scatter(df_r[x], df_r[y], s=140, alpha=0.8)
            for _, r in df_r.iterrows():
                ax.annotate(r["Model"], (r[x] + 0.02 * df_r[x].max(), r[y] * 0.99), fontsize=9)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_title(f"{y} vs {x}")
            ax.grid(True, linestyle="--", alpha=0.3)
            st.pyplot(fig)

        st.markdown("---")
        st.markdown("### 🔍 Cosine Similarity (TF-IDF)")
        s1 = st.text_area("Text A", placeholder="Enter text A...", key="cosA", height=90)
        s2 = st.text_area("Text B", placeholder="Enter text B...", key="cosB", height=90)
        if st.button("Compare texts"):
            compute_and_display_similarity(s1, s2)

        # Animated neon sine preview (small)
        st.markdown("---")
        st.markdown("### ✨ Neon Preview")
        x = np.linspace(0, 4 * np.pi, 160)
        frames = []
        for phase in np.linspace(0, 2 * np.pi, 36):
            yvals = np.sin(x + phase) * (0.7 + 0.3 * np.sin(phase * 2))
            frames.append(go.Frame(data=[go.Scatter(x=x, y=yvals, mode="lines", line=dict(color="#00ffd1", width=2))]))

        mini_fig = go.Figure(
            data=[go.Scatter(x=x, y=np.sin(x), mode="lines", line=dict(color="#00ffd1", width=2))],
            layout=go.Layout(template="plotly_dark", margin=dict(l=10, r=10, t=20, b=10), height=220,
                            xaxis=dict(showgrid=False, visible=False), yaxis=dict(showgrid=False, visible=False),
                            updatemenus=[{
                                "buttons": [
                                    {"args": [None, {"frame": {"duration": 60, "redraw": True}, "fromcurrent": True}],
                                     "label": "Play", "method": "animate"}
                                ],
                                "direction": "left",
                                "showactive": False,
                                "type": "buttons",
                                "x": 0.0,
                                "y": 0
                            }]),
            frames=frames
        )
        st.plotly_chart(mini_fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Footer ----------
    st.markdown('<div class="app-footer">Built by Neha</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    run_app()  
