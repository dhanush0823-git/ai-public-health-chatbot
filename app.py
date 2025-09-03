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
st.set_page_config(page_title="SwasthyaBot", page_icon="🩺", layout="wide")
APP_TITLE = "🩺 SwasthyaBot — AI Health Chatbot"

# ---------------------------
# SESSION STATE INIT
# ---------------------------
if "user_name" not in st.session_state:
    st.session_state.user_name = "Guest"

if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, Any]] = []

# ---------------------------
# DATASET URLS
# ---------------------------
FAQ_URL = "https://raw.githubusercontent.com/dhanush0823-git/ai-public-health-chatbot/refs/heads/main/health_faq.csv"
VACCINE_URL = "https://raw.githubusercontent.com/dhanush0823-git/ai-public-health-chatbot/refs/heads/main/vaccination.csv"
OUTBREAK_URL = "https://raw.githubusercontent.com/dhanush0823-git/ai-public-health-chatbot/refs/heads/main/outbreak.csv"

# ---------------------------
# CSV LOADER (silent)
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
    df["question"] = df["question"].astype(str).str.strip()
    df["answer"] = df["answer"].astype(str).str.strip()
    df["category"] = category
    return df[["question", "answer", "category"]]

faq_df = load_dataset(FAQ_URL, "Health FAQ")
vaccine_df = load_dataset(VACCINE_URL, "Vaccination")
outbreak_df = load_dataset(OUTBREAK_URL, "Outbreak")

kb = pd.concat([faq_df, vaccine_df, outbreak_df], ignore_index=True).dropna(subset=["question", "answer"])
kb = kb.reset_index(drop=True)

# ---------------------------
# EMBEDDINGS
# ---------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder, kb_embeddings = None, None
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
        return "⚠️ Knowledge base not available. Please check the dataset files."
    try:
        q_emb = embedder.encode(query, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(q_emb, kb_embeddings)[0]
        best_idx = int(torch.argmax(scores).item())
        best_score = float(scores[best_idx])
        if best_score < 0.35:
            return "🤔 I couldn't find a confident answer. Please rephrase your question or consult a health worker."
        return str(kb.loc[best_idx, "answer"])
    except Exception as e:
        return f"Error: {e}"

# ---------------------------
# UI LAYOUT
# ---------------------------
# Sidebar
with st.sidebar:
    st.title("⚙️ Menu")
    st.subheader("Patient Info")
    st.session_state.user_name = st.text_input("Name", value=st.session_state.user_name)
    if st.button("🗑 Clear chat"):
        st.session_state.history = []
        st.experimental_rerun()

# Main Title
st.markdown(f"<h1 style='text-align:center;'>{APP_TITLE}</h1>", unsafe_allow_html=True)
st.divider()

# ---------------------------
# Chat Input
# ---------------------------
user_message = st.chat_input("Ask about health, vaccination, or outbreaks...")

if user_message:
    answer_text = get_answer_from_kb(user_message)
    st.session_state.history.append({
        "user": user_message,
        "bot": answer_text,
        "name": st.session_state.user_name or "Guest",
        "time": datetime.utcnow().isoformat()
    })

# ---------------------------
# Render Chat (like ChatGPT)
# ---------------------------
for entry in st.session_state.history:
    name = entry.get("name", "Guest")
    user_txt = entry.get("user", "")
    bot_txt = entry.get("bot", "")

    with st.chat_message("user"):
        st.markdown(f"**{name}:** {user_txt}")

    with st.chat_message("assistant"):
        st.markdown(bot_txt)
