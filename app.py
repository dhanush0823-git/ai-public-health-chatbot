# app.py
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import MarianMTModel, MarianTokenizer
import torch
from typing import Tuple, Optional

# ---------------------------
# CONFIG / DATA URLS
# ---------------------------
st.set_page_config(page_title="SwasthyaBot", layout="wide")
st.title("🩺 D's Bot – AI Health Chatbot")

# Raw GitHub links (your repo)
FAQ_URL = "https://raw.githubusercontent.com/dhanush0823-git/ai-public-health-chatbot/refs/heads/main/health_faq.csv"
VACCINE_URL = "https://raw.githubusercontent.com/dhanush0823-git/ai-public-health-chatbot/refs/heads/main/vaccination.csv"
OUTBREAK_URL = "https://raw.githubusercontent.com/dhanush0823-git/ai-public-health-chatbot/refs/heads/main/outbreak.csv"

# ---------------------------
# DATA LOADER (robust)
# ---------------------------
@st.cache_data
def load_dataset(url: str, category: str) -> pd.DataFrame:
    """
    Robust loader: uses python engine, utf-8, skips bad lines.
    Adds 'category' column. Returns empty DataFrame on failure.
    """
    try:
        # try common separators and fallbacks if needed
        df = pd.read_csv(url, engine="python", encoding="utf-8", on_bad_lines="skip")
        # If CSV actually uses semicolon, try that as fallback if only 1 column
        if df.shape[1] == 1 and ";" in df.columns[0]:
            df = pd.read_csv(url, engine="python", encoding="utf-8", sep=";", on_bad_lines="skip")
        # Normalize column names (lowercase)
        df.columns = [c.strip().lower() for c in df.columns]
        # Ensure we have 'question' and 'answer' columns (attempt to guess)
        if "question" not in df.columns or "answer" not in df.columns:
            # try common alternatives
            possible_q = [c for c in df.columns if "question" in c or "q" == c or "query" in c]
            possible_a = [c for c in df.columns if "answer" in c or "ans" in c or "response" in c]
            if possible_q and possible_a:
                df = df.rename(columns={possible_q[0]: "question", possible_a[0]: "answer"})
            else:
                # try to split single-column CSV (question|answer) by first comma
                if df.shape[1] == 1:
                    col0 = df.columns[0]
                    splits = df[col0].astype(str).str.split(",", n=1, expand=True)
                    if splits.shape[1] == 2:
                        df = pd.DataFrame({"question": splits[0].str.strip(), "answer": splits[1].str.strip()})
                    else:
                        st.warning(f"{category}: Could not find question/answer columns; file has unexpected format.")
                        return pd.DataFrame(columns=["question", "answer", "category"])
                else:
                    st.warning(f"{category}: Could not find question/answer columns; file columns: {df.columns.tolist()}")
                    return pd.DataFrame(columns=["question", "answer", "category"])
        # Keep only question & answer (and other useful columns)
        df = df.rename(columns={c: c for c in df.columns})
        df = df[["question", "answer"] + [c for c in df.columns if c not in ["question", "answer"]]]
        df["question"] = df["question"].astype(str).str.strip()
        df["answer"] = df["answer"].astype(str).str.strip()
        df["category"] = category
        st.info(f"{category} dataset loaded ({len(df)} rows).")
        return df
    except Exception as e:
        st.error(f"Failed to load {category} dataset: {e}")
        return pd.DataFrame(columns=["question", "answer", "category"])

# Load datasets
faq_df = load_dataset(FAQ_URL, "Health FAQ")
vaccine_df = load_dataset(VACCINE_URL, "Vaccination")
outbreak_df = load_dataset(OUTBREAK_URL, "Outbreak Alert")

# Merge into KB
kb = pd.concat([faq_df, vaccine_df, outbreak_df], ignore_index=True)
# drop empty rows
kb = kb.dropna(subset=["question", "answer"]).reset_index(drop=True)

if kb.empty:
    st.warning("Knowledge base is empty. Please check your CSV files in the repo.")
else:
    st.success(f"Knowledge base ready with {len(kb)} items.")

# ---------------------------
# EMBEDDING MODEL (cached)
# ---------------------------
@st.cache_resource
def get_embedder():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        raise

embedder = None
kb_embeddings = None
if not kb.empty:
    embedder = get_embedder()
    # compute embeddings once and cache
    @st.cache_data
    def compute_kb_embeddings(texts):
        return embedder.encode(texts, convert_to_tensor=True)
    kb_embeddings = compute_kb_embeddings(kb["question"].tolist())

# ---------------------------
# TRANSLATION UTILITIES (cached translator instances)
# ---------------------------
TRANSLATOR_CACHE = {}

def load_translator_pair(src: str, tgt: str) -> Optional[Tuple[MarianTokenizer, MarianMTModel]]:
    key = f"{src}-{tgt}"
    if key in TRANSLATOR_CACHE:
        return TRANSLATOR_CACHE[key]
    # Only load common pairs or try best-effort; many models may not exist
    model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        TRANSLATOR_CACHE[key] = (tokenizer, model)
        return tokenizer, model
    except Exception:
        # silent fallback: no translator available
        return None

def translate_text(text: str, src: str, tgt: str) -> str:
    if not text or src == tgt:
        return text
    pair = load_translator_pair(src, tgt)
    if pair is None:
        # fallback: return original text (no translation)
        return text
    tokenizer, model = pair
    try:
        inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=512)
        gen = model.generate(**inputs, max_length=512)
        out = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        return out
    except Exception:
        return text

# ---------------------------
# SEARCH / ANSWER
# ---------------------------
def answer_from_kb(user_query: str, user_lang: str = "en") -> Tuple[str, str, str]:
    """
    Returns (answer_text, matched_question, category)
    """
    if kb.empty:
        return ("Sorry, knowledge base is empty.", "", "")
    # 1. translate to English for processing
    query_en = translate_text(user_query, user_lang, "en")
    # 2. embed
    try:
        q_emb = embedder.encode(query_en, convert_to_tensor=True)
    except Exception as e:
        return (f"Embedding failed: {e}", "", "")
    # 3. similarity search (cosine)
    try:
        scores = util.pytorch_cos_sim(q_emb, kb_embeddings)[0]
        best_idx = int(torch.argmax(scores).item())
        best_score = float(scores[best_idx])
        matched_q = kb.loc[best_idx, "question"]
        matched_a = kb.loc[best_idx, "answer"]
        matched_cat = kb.loc[best_idx, "category"] if "category" in kb.columns else ""
        # If score too low, be honest
        if best_score < 0.35:
            reply_en = "I couldn't find a confident answer in my knowledge base. Please rephrase or consult a health worker."
        else:
            reply_en = matched_a
        # translate back to user language
        reply_user = translate_text(reply_en, "en", user_lang)
        return (reply_user, matched_q, matched_cat)
    except Exception as e:
        return (f"Search failed: {e}", "", "")

# ---------------------------
# STREAMLIT UI (chat)
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []

st.sidebar.header("Settings")
lang = st.sidebar.selectbox("Language", ["en", "hi", "ta", "te", "mr", "bn"], index=0)
st.sidebar.markdown("Datasets loaded:")
st.sidebar.write(f"- Health FAQ: {len(faq_df)} rows")
st.sidebar.write(f"- Vaccination: {len(vaccine_df)} rows")
st.sidebar.write(f"- Outbreak: {len(outbreak_df)} rows")

st.header("Chat with SwasthyaBot")
user_name = st.text_input("Your name (optional)", value="Guest")
user_input = st.text_area("Ask about health, vaccination or outbreaks", height=120)

if st.button("Ask"):
    if not user_input.strip():
        st.warning("Please type a question.")
    else:
        with st.spinner("Thinking..."):
            ans, matched_q, cat = answer_from_kb(user_input, user_lang=lang)
            st.session_state.history.append({"user": user_input, "answer": ans, "matched_q": matched_q, "category": cat, "user_name": user_name})

# show chat history (most recent first)
for entry in reversed(st.session_state.history):
    st.markdown(f"**You:** {entry['user']}")
    st.markdown(f"**SwasthyaBot:** {entry['answer']}")
    if entry.get("matched_q"):
        st.caption(f"Matched from: {entry['matched_q']}  |  Category: {entry.get('category','')}")
    st.write("---")

# Small footer / instructions
st.info("If dataset columns or CSV format are unexpected, edit CSVs in the repo and ensure 'question' and 'answer' columns exist.")
