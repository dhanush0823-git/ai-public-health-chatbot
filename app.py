# app.py
import streamlit as st
import pandas as pd
import torch
from datetime import datetime
from typing import List, Dict, Any

# Embeddings
from sentence_transformers import SentenceTransformer, util

# HuggingFace LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="🩺 Crafter's — AI Health Chatbot", layout="centered")
APP_TITLE = "🩺 Crafter's — AI Health Chatbot"

# ---------------------------
# SESSION STATE INIT
# ---------------------------
if "user_name" not in st.session_state:
    st.session_state.user_name = "Guest"

if "history" not in st.session_state:
    st.session_state.history = []

if "mode" not in st.session_state:
    st.session_state.mode = "Dataset"

# ---------------------------
# DATASET LOADER
# ---------------------------
FAQ_URL = "https://raw.githubusercontent.com/dhanush0823-git/ai-public-health-chatbot/refs/heads/main/health_faq.csv"
VACCINE_URL = "https://raw.githubusercontent.com/dhanush0823-git/ai-public-health-chatbot/refs/heads/main/vaccination.csv"
OUTBREAK_URL = "https://raw.githubusercontent.com/dhanush0823-git/ai-public-health-chatbot/refs/heads/main/outbreak.csv"

@st.cache_data
def load_dataset(url: str, category: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url, encoding="utf-8", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(url, encoding="latin1", on_bad_lines="skip")
    df.columns = [c.strip().lower() for c in df.columns]
    if "question" not in df.columns or "answer" not in df.columns:
        return pd.DataFrame(columns=["question", "answer", "category"])
    df["category"] = category
    return df[["question", "answer", "category"]]

faq_df = load_dataset(FAQ_URL, "Health FAQ")
vaccine_df = load_dataset(VACCINE_URL, "Vaccination")
outbreak_df = load_dataset(OUTBREAK_URL, "Outbreak")

kb = pd.concat([faq_df, vaccine_df, outbreak_df], ignore_index=True).dropna(subset=["question", "answer"])

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

def get_dataset_answer(query: str) -> str:
    if kb.empty or kb_embeddings is None:
        return "⚠️ Knowledge base not available."
    try:
        q_emb = embedder.encode(query, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(q_emb, kb_embeddings)[0]
        best_idx = int(torch.argmax(scores).item())
        best_score = float(scores[best_idx])
        if best_score < 0.35:
            return "❓ I couldn’t find a confident answer. Please rephrase your question."
        return str(kb.loc[best_idx, "answer"])
    except Exception as e:
        return f"⚠️ Error: {e}"

# ---------------------------
# LLM MODE (Optional, GPU needed)
# ---------------------------
@st.cache_resource
def load_llm():
    model_name = "distilgpt2"  # ✅ lightweight fallback; replace with Vicuna if you have GPU
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

llm_pipeline = None
def get_llm_answer(query: str, context: List[Dict[str, str]]) -> str:
    global llm_pipeline
    if llm_pipeline is None:
        llm_pipeline = load_llm()

    structured_context = "\n".join(
        [f"User: {entry['user']}\nAssistant: {entry['bot']}" for entry in context[-5:]]
    )

    prompt = f"""
You are a helpful and safe medical assistant.
Use the context if relevant.

Context:
{structured_context}

User: {query}
Assistant:"""

    try:
        response = llm_pipeline(prompt, max_length=200, do_sample=True, temperature=0.7)
        return response[0]["generated_text"].replace(prompt, "").strip()
    except Exception as e:
        return f"⚠️ LLM error: {e}"

# ---------------------------
# SIDEBAR
# ---------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state.user_name = st.text_input("Your Name", value=st.session_state.user_name)
    st.session_state.mode = st.radio("Answering Mode:", ["Dataset", "LLM"])
    if st.button("🗑 Clear chat"):
        st.session_state.history = []
        st.experimental_rerun()

# ---------------------------
# UI HEADER
# ---------------------------
st.markdown(f"<h1 style='text-align:center'>{APP_TITLE}</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------
# CHAT INPUT
# ---------------------------
user_message = st.chat_input("Ask about health, vaccination, or outbreaks...")

if user_message:
    if st.session_state.mode == "Dataset":
        bot_reply = get_dataset_answer(user_message)
    else:
        bot_reply = get_llm_answer(user_message, st.session_state.history)

    st.session_state.history.append({
        "user": user_message,
        "bot": bot_reply,
        "time": datetime.utcnow().isoformat()
    })

# ---------------------------
# DISPLAY CHAT
# ---------------------------
for entry in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(f"**{st.session_state.user_name}:** {entry['user']}")
    with st.chat_message("assistant"):
        st.markdown(entry['bot'])
