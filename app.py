# app.py
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import MarianMTModel, MarianTokenizer
import torch

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="SwasthyaBot", layout="centered")

# ---------------------------
# PATIENT NAME
# ---------------------------
if "user_name" not in st.session_state:
    st.session_state.user_name = "Guest"

st.sidebar.header("Patient Info")
st.session_state.user_name = st.sidebar.text_input("Enter your name", value=st.session_state.user_name)

# ---------------------------
# HEADER
# ---------------------------
st.markdown(
    f"""
    <div style="text-align: center; padding: 1rem; border-radius: 12px; background: linear-gradient(90deg, #1E3C72, #2A5298); color: white;">
        <h1 style="margin-bottom: 0;">🩺 SwasthyaBot</h1>
        <p style="margin-top: 0;">AI Health Assistant for {st.session_state.user_name}</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# LOAD DATASETS (silent)
# ---------------------------
@st.cache_data
def load_dataset(url, category):
    try:
        df = pd.read_csv(url, engine="python", encoding="utf-8", on_bad_lines="skip")
        df.columns = [c.strip().lower() for c in df.columns]
        if "question" not in df.columns or "answer" not in df.columns:
            return pd.DataFrame(columns=["question", "answer", "category"])
        df["category"] = category
        return df
    except:
        return pd.DataFrame(columns=["question", "answer", "category"])

FAQ_URL = "https://raw.githubusercontent.com/dhanush0823-git/ai-public-health-chatbot/refs/heads/main/health_faq.csv"
VACCINE_URL = "https://raw.githubusercontent.com/dhanush0823-git/ai-public-health-chatbot/refs/heads/main/vaccination.csv"
OUTBREAK_URL = "https://raw.githubusercontent.com/dhanush0823-git/ai-public-health-chatbot/refs/heads/main/outbreak.csv"

faq_df = load_dataset(FAQ_URL, "Health FAQ")
vaccine_df = load_dataset(VACCINE_URL, "Vaccination")
outbreak_df = load_dataset(OUTBREAK_URL, "Outbreak")
kb = pd.concat([faq_df, vaccine_df, outbreak_df], ignore_index=True).dropna(subset=["question", "answer"])

# ---------------------------
# EMBEDDINGS
# ---------------------------
@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = get_embedder()
kb_embeddings = embedder.encode(kb["question"].tolist(), convert_to_tensor=True) if not kb.empty else None

# ---------------------------
# CHAT ENGINE
# ---------------------------
def get_answer(query):
    if kb.empty or kb_embeddings is None:
        return "⚠️ Knowledge base is empty."
    q_emb = embedder.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(q_emb, kb_embeddings)[0]
    best_idx = int(torch.argmax(scores).item())
    best_score = float(scores[best_idx])
    if best_score < 0.35:
        return "I couldn't find a confident answer. Please consult a health worker."
    return kb.loc[best_idx, "answer"]

# ---------------------------
# CHAT UI
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("### 💬 Chat with SwasthyaBot")
user_input = st.chat_input("Ask about health, vaccination or outbreaks...")

if user_input:
    answer = get_answer(user_input)
    st.session_state.history.append({"user": user_input, "bot": answer})

# Render chat history
for chat in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(f"**{st.session_state.user_name}:** {chat['user']}")
    with st.chat_message("assistant"):
        st.markdown(chat["bot"])
