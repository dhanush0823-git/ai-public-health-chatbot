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
st.set_page_config(page_title="Crafter's", layout="centered")
APP_TITLE = "🩺 Crafter's — AI Health Chatbot"

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
                continue
            normalized.append({"user": str(user_text), "bot": str(bot_text), "name": name, "time": ts})
        except Exception:
            continue
    st.session_state.history = normalized

_normalize_history()

# ---------------------------
# DATASET URLS
# ---------------------------
FAQ_URL = "https://raw.githubusercontent.com/dhanush0823-git/ai-public-health-chatbot/refs/heads/main/healthfaq.csv"
VACCINE_URL = "https://raw.githubusercontent.com/dhanush0823-git/ai-public-health-chatbot/refs/heads/main/vaccination.csv"
OUTBREAK_URL = "https://raw.githubusercontent.com/dhanush0823-git/ai-public-health-chatbot/refs/heads/main/outbreak.csv"

# ---------------------------
# ROBUST CSV LOADER
# ---------------------------
@st.cache_data
def load_dataset(url: str, category: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url, engine="python", encoding="utf-8", on_bad_lines="skip")
    except Exception:
        try:
            df = pd.read_csv(url, engine="python", encoding="latin1", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame(columns=["question", "answer", "category"])
    df.columns = [c.strip().lower() for c in df.columns]
    if "question" not in df.columns or "answer" not in df.columns:
        return pd.DataFrame(columns=["question", "answer", "category"])
    df["category"] = category
    return df[["question", "answer", "category"]]

faq_df = load_dataset(FAQ_URL, "Health FAQ")
vaccine_df = load_dataset(VACCINE_URL, "Vaccination")
outbreak_df = load_dataset(OUTBREAK_URL, "Outbreak")

kb = pd.concat([faq_df, vaccine_df, outbreak_df], ignore_index=True).dropna(subset=["question", "answer"])
kb = kb.reset_index(drop=True)

# ---------------------------
# EMBEDDING MODEL
# ---------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = None
kb_embeddings = None
if not kb.empty:
    embedder = load_embedder()
    @st.cache_data
    def compute_embeddings(texts):
        return embedder.encode(texts, convert_to_tensor=True)
    kb_embeddings = compute_embeddings(kb["question"].tolist())

# ---------------------------
# ANSWER FUNCTION
# ---------------------------
def get_answer_from_kb(query: str) -> str:
    if kb.empty or kb_embeddings is None:
        return "⚠️ Knowledge base not available. Please check the dataset."
    try:
        q_emb = embedder.encode(query, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(q_emb, kb_embeddings)[0]
        best_idx = int(torch.argmax(scores).item())
        best_score = float(scores[best_idx])
        if best_score < 0.35:
            return "❓ I couldn’t find a confident answer. Please rephrase your question."
        return str(kb.loc[best_idx, "answer"])
    except Exception as e:
        return f"Error retrieving answer: {e}"

# ---------------------------
# UI HEADER + SIDEBAR
# ---------------------------
st.markdown(f"<h1 style='text-align:center'>{APP_TITLE}</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

with st.sidebar:
    st.header("👤 Patient Info")
    st.session_state.user_name = st.text_input("Your Name", value=st.session_state.user_name)
    if st.button("🗑 Clear chat"):
        st.session_state.history = []
        st.rerun()

# ---------------------------
# Chat Input
# ---------------------------
user_message = None
try:
    user_message = st.chat_input("Ask about health, vaccination, or outbreaks...")
except Exception:
    user_message = st.text_input("Ask about health, vaccination, or outbreaks:")

if user_message:
    name_at_time = st.session_state.user_name or "Guest"
    answer_text = get_answer_from_kb(user_message)
    entry = {"user": str(user_message), "bot": str(answer_text), "name": name_at_time, "time": datetime.utcnow().isoformat()}
    st.session_state.history.append(entry)

# ---------------------------
# Render chat / Welcome page
# ---------------------------
def render_history():
    if not st.session_state.history:
        st.markdown(
            f"""
            <div style='text-align:center; padding:20px;'>
                <h3>👋 Welcome {st.session_state.user_name}!</h3>
                <p>I’m <b>Crafter's</b>,AI assistant for:</p>
                <ul style='text-align:left; max-width:500px; margin:auto;'>
                    <li>🩺 Health FAQs</li>
                    <li>💉 Vaccination details</li>
                    <li>🚨 Outbreak alerts</li>
                </ul>
                <p>Type your question below to begin the conversation.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for entry in st.session_state.history:
        if not isinstance(entry, dict):
            continue
        user_txt = entry.get("user", "")
        bot_txt = entry.get("bot", "⚠️ No answer available.")
        name = entry.get("name", st.session_state.user_name or "Guest")
        try:
            with st.chat_message("user"):
                st.markdown(f"**{name}:** {user_txt}")
        except Exception:
            st.markdown(f"**{name}:** {user_txt}")
        try:
            with st.chat_message("assistant"):
                st.markdown(bot_txt)
        except Exception:
            st.markdown(bot_txt)

render_history()
