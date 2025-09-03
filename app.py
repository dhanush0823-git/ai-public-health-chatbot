# app.py
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from datetime import datetime
from typing import List, Dict, Any

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="SwasthyaBot", layout="centered")
APP_TITLE = "🩺 SwasthyaBot — AI Health Chatbot"

# ---------------------------
# SESSION STATE INIT & NORMALIZE
# ---------------------------
if "user_name" not in st.session_state:
    st.session_state.user_name = "Guest"

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {"user":..., "bot":..., "name":..., "time":...}

def _normalize_history():
    """Make sure history entries are dicts with at least 'user' and 'bot' keys."""
    normalized: List[Dict[str, Any]] = []
    for entry in st.session_state.history:
        try:
            if isinstance(entry, dict):
                user_text = entry.get("user", "")
                bot_text = entry.get("bot", entry.get("answer", ""))
                name = entry.get("name", st.session_state.user_name)
                ts = entry.get("time", datetime.utcnow().isoformat())
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                user_text = entry[0]
                bot_text = entry[1]
                name = st.session_state.user_name
                ts = datetime.utcnow().isoformat()
            else:
                # unknown format -> skip
                continue
            normalized.append({"user": str(user_text), "bot": str(bot_text), "name": name, "time": ts})
        except Exception:
            # skip malformed entries
            continue
    st.session_state.history = normalized

_normalize_history()

# ---------------------------
# DATASET URLS (your raw GitHub links)
# ---------------------------
FAQ_URL = "https://raw.githubusercontent.com/dhanush0823-git/ai-public-health-chatbot/refs/heads/main/health_faq.csv"
VACCINE_URL = "https://raw.githubusercontent.com/dhanush0823-git/ai-public-health-chatbot/refs/heads/main/vaccination.csv"
OUTBREAK_URL = "https://raw.githubusercontent.com/dhanush0823-git/ai-public-health-chatbot/refs/heads/main/outbreak.csv"

# ---------------------------
# ROBUST CSV LOADER (silent)
# ---------------------------
@st.cache_data
def load_dataset(url: str, category: str) -> pd.DataFrame:
    """Load CSV from raw URL with fallbacks; returns DataFrame with 'question' & 'answer' columns or empty df."""
    try:
        df = pd.read_csv(url, engine="python", encoding="utf-8", on_bad_lines="skip")
    except Exception:
        try:
            df = pd.read_csv(url, engine="python", encoding="latin1", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame(columns=["question", "answer", "category"])
    # Normalize columns
    df.columns = [c.strip().lower() for c in df.columns]
    if "question" not in df.columns or "answer" not in df.columns:
        # try to guess common alternative names
        q_col = next((c for c in df.columns if "question" in c or c == "q" or "query" in c), None)
        a_col = next((c for c in df.columns if "answer" in c or c == "ans" or "response" in c), None)
        if q_col and a_col:
            df = df.rename(columns={q_col: "question", a_col: "answer"})
        else:
            # if single-column, attempt split on first comma
            if df.shape[1] == 1:
                col0 = df.columns[0]
                split = df[col0].astype(str).str.split(",", n=1, expand=True)
                if split.shape[1] == 2:
                    df = pd.DataFrame({"question": split[0].str.strip(), "answer": split[1].str.strip()})
                else:
                    return pd.DataFrame(columns=["question", "answer", "category"])
            else:
                return pd.DataFrame(columns=["question", "answer", "category"])
    df = df.rename(columns={c: c for c in df.columns})
    df = df[["question", "answer"] + [c for c in df.columns if c not in ("question", "answer")]]
    df["question"] = df["question"].astype(str).str.strip()
    df["answer"] = df["answer"].astype(str).str.strip()
    df["category"] = category
    return df

# load silently (no UI success messages)
faq_df = load_dataset(FAQ_URL, "Health FAQ")
vaccine_df = load_dataset(VACCINE_URL, "Vaccination")
outbreak_df = load_dataset(OUTBREAK_URL, "Outbreak")

kb = pd.concat([faq_df, vaccine_df, outbreak_df], ignore_index=True).dropna(subset=["question", "answer"])
kb = kb.reset_index(drop=True)

# ---------------------------
# EMBEDDING MODEL (cached)
# ---------------------------
@st.cache_resource
def load_embedder():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        # fail loudly so user knows something wrong with model load
        raise

embedder = None
kb_embeddings = None
if not kb.empty:
    embedder = load_embedder()
    @st.cache_data
    def compute_embeddings(texts):
        return embedder.encode(texts, convert_to_tensor=True)
    kb_embeddings = compute_embeddings(kb["question"].tolist())

# ---------------------------
# ANSWER FUNCTION (robust)
# ---------------------------
def get_answer_from_kb(query: str) -> str:
    if kb.empty or kb_embeddings is None:
        return "Sorry — knowledge base not available. Please check the dataset files."
    try:
        q_emb = embedder.encode(query, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(q_emb, kb_embeddings)[0]
        best_idx = int(torch.argmax(scores).item())
        best_score = float(scores[best_idx])
        if best_score < 0.35:
            return "I couldn't find a confident answer. Please rephrase your question or consult a health worker."
        return str(kb.loc[best_idx, "answer"])
    except Exception as e:
        return f"Error retrieving answer: {e}"

# ---------------------------
# UI: Header + Sidebar (name entry)
# ---------------------------
st.markdown(f"<h1 style='text-align:center'>{APP_TITLE}</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Patient")
    st.session_state.user_name = st.text_input("Name", value=st.session_state.user_name)
    if st.button("Clear chat"):
        st.session_state.history = []
        st.experimental_rerun()

# ---------------------------
# Chat input (ChatGPT style)
# ---------------------------
st.markdown("### 💬 Chat")

# Use st.chat_input if available, otherwise fallback to text_input
user_message = None
try:
    user_message = st.chat_input("Ask about health, vaccination, or outbreaks...")
except Exception:
    user_message = st.text_input("Ask about health, vaccination, or outbreaks:")

if user_message:
    name_at_time = st.session_state.user_name or "Guest"
    answer_text = get_answer_from_kb(user_message)
    entry = {
        "user": str(user_message),
        "bot": str(answer_text),
        "name": name_at_time,
        "time": datetime.utcnow().isoformat()
    }
    st.session_state.history.append(entry)

# ---------------------------
# Render chat (most recent last)
# ---------------------------
# Use chat_message if available for nicer UI
def render_history():
    for entry in st.session_state.history:
        # ensure dict safety
        if not isinstance(entry, dict):
            continue
        user_txt = entry.get("user", "")
        bot_txt = entry.get("bot", "Sorry, no answer available.")
        name = entry.get("name", st.session_state.user_name or "Guest")
        # display user
        try:
            with st.chat_message("user"):
                st.markdown(f"**{name}:** {user_txt}")
        except Exception:
            st.markdown(f"**{name}:** {user_txt}")
        # display bot
        try:
            with st.chat_message("assistant"):
                st.markdown(bot_txt)
        except Exception:
            st.markdown(bot_txt)

render_history()
