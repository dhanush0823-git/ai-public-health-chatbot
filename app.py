"""
Crafter's — AI Health Chatbot
Gemini-powered public-health assistant with optional CSV knowledge-base retrieval.

Environment:
    Gemini_Api=your_gemini_api_key

Run:
    streamlit run app.py
"""

import os
import html
from datetime import datetime, timezone
from io import StringIO
from typing import Optional

import pandas as pd
import requests
import streamlit as st
from google import genai
from google.genai import types


# -------------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Crafter's | AI Health Assistant",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

APP_TITLE = "Crafter's Health Assistant"
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

# Use the raw GitHub URL for CSV content (not the GitHub HTML page).
DATASET_URLS = {
    "Health FAQ": "https://raw.githubusercontent.com/dhanush0823-git/ai-public-health-chatbot/main/health_faq.csv",
    "Vaccination": "https://raw.githubusercontent.com/dhanush0823-git/ai-public-health-chatbot/main/vaccination.csv",
    "Outbreak": "https://raw.githubusercontent.com/dhanush0823-git/ai-public-health-chatbot/main/outbreak.csv",
}

SYSTEM_INSTRUCTION = """
You are Crafter's Health Assistant, a careful, empathetic public-health information assistant.
Your role is to provide general health education, explain vaccination and public-health topics,
and summarize relevant information from the supplied knowledge-base context.

Safety requirements:
- You are not a doctor and must not claim to diagnose, prescribe, or replace professional care.
- Do not make up medical facts, sources, statistics, vaccination schedules, or outbreak updates.
- If the knowledge-base context does not support a specific local/current claim, say so clearly.
- Give practical, general information and encourage the user to consult a qualified healthcare
  professional for personal medical decisions.
- Never recommend starting, stopping, or changing prescription medication without a clinician.
- If symptoms suggest a possible emergency (for example, severe difficulty breathing, chest pain,
  signs of stroke, severe bleeding, or loss of consciousness), advise contacting local emergency
  services immediately.
- Ask a short clarifying question when important details are missing; do not overstate certainty.
- Keep answers clear, compassionate, and reasonably concise. Use bullets where they improve clarity.
- Do not repeat personal or sensitive information unnecessarily.
"""


# -------------------------------------------------------------------
# SESSION STATE
# -------------------------------------------------------------------
def init_session_state() -> None:
    defaults = {
        "user_name": "Guest",
        "history": [],
        "use_knowledge_base": True,
        "show_timestamps": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# -------------------------------------------------------------------
# DATASET LOADING
# -------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Loading public-health knowledge base...")
def load_dataset(url: str, category: str) -> pd.DataFrame:
    """Load one CSV safely. An unavailable dataset won't stop the app."""
    empty = pd.DataFrame(columns=["question", "answer", "category"])

    try:
        response = requests.get(url, timeout=(5, 20))
        response.raise_for_status()
        df = pd.read_csv(
            StringIO(response.content.decode("utf-8-sig", errors="replace")),
            on_bad_lines="skip",
        )
    except (requests.RequestException, pd.errors.ParserError, UnicodeDecodeError):
        return empty
    except Exception:
        return empty

    df.columns = [str(column).strip().lower() for column in df.columns]
    if not {"question", "answer"}.issubset(df.columns):
        return empty

    df = df[["question", "answer"]].copy()
    df["question"] = df["question"].fillna("").astype(str).str.strip()
    df["answer"] = df["answer"].fillna("").astype(str).str.strip()
    df = df[(df["question"] != "") & (df["answer"] != "")]
    df["category"] = category
    return df[["question", "answer", "category"]]


@st.cache_data(ttl=3600, show_spinner=False)
def load_knowledge_base() -> pd.DataFrame:
    frames = [load_dataset(url, category) for category, url in DATASET_URLS.items()]
    frames = [frame for frame in frames if not frame.empty]

    if not frames:
        return pd.DataFrame(columns=["question", "answer", "category"])

    kb = pd.concat(frames, ignore_index=True)
    kb = kb.drop_duplicates(subset=["question", "answer"]).reset_index(drop=True)
    return kb


kb = load_knowledge_base()


def retrieve_context(query: str, max_items: int = 4) -> str:
    """
    Lightweight lexical retrieval that avoids downloading a local embedding model.
    It uses question-token overlap as a small context selector; Gemini writes the answer.
    """
    if kb.empty:
        return "The CSV knowledge base is currently unavailable."

    query_terms = {
        token.lower().strip(".,!?;:()[]{}'\"")
        for token in query.split()
        if len(token.strip(".,!?;:()[]{}'\"")) > 2
    }
    if not query_terms:
        return "No directly matched knowledge-base entries were found."

    ranked = []
    for _, row in kb.iterrows():
        question = str(row["question"])
        answer = str(row["answer"])
        question_terms = {
            token.lower().strip(".,!?;:()[]{}'\"")
            for token in question.split()
            if len(token.strip(".,!?;:()[]{}'\"")) > 2
        }
        answer_terms = {
            token.lower().strip(".,!?;:()[]{}'\"")
            for token in answer.split()
            if len(token.strip(".,!?;:()[]{}'\"")) > 2
        }

        if not question_terms:
            continue

        question_overlap = len(query_terms & question_terms) / max(len(query_terms), 1)
        answer_overlap = len(query_terms & answer_terms) / max(len(query_terms), 1)
        score = (0.75 * question_overlap) + (0.25 * answer_overlap)

        if score > 0:
            ranked.append((score, question, answer, str(row["category"])))

    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = ranked[:max_items]

    if not selected:
        return "No directly matched knowledge-base entries were found."

    return "\n\n".join(
        f"Category: {category}\nQuestion: {question}\nKnowledge-base answer: {answer}"
        for _, question, answer, category in selected
    )


# -------------------------------------------------------------------
# GEMINI CLIENT AND RESPONSE
# -------------------------------------------------------------------
@st.cache_resource
def get_gemini_client(api_key: str):
    return genai.Client(api_key=api_key)


def make_gemini_contents(
    user_message: str,
    chat_history: list,
    context_text: str,
) -> list:
    """Convert recent app history into Gemini's multi-turn content format."""
    contents = []

    # Keep a short recent window to limit token use and retain useful context.
    for item in chat_history[-8:]:
        previous_user = str(item.get("user", "")).strip()
        previous_bot = str(item.get("bot", "")).strip()

        if previous_user:
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=previous_user)],
                )
            )
        if previous_bot:
            contents.append(
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=previous_bot)],
                )
            )

    final_prompt = (
        f"Knowledge-base context (may be incomplete; do not treat it as proof if irrelevant):\n"
        f"{context_text}\n\n"
        f"User question:\n{user_message}\n\n"
        "Answer the user's latest question. If the supplied context is relevant, use it. "
        "If it is not relevant or does not contain the answer, say what is uncertain instead "
        "of inventing details."
    )
    contents.append(
        types.Content(role="user", parts=[types.Part.from_text(text=final_prompt)])
    )
    return contents


def generate_answer(user_message: str, chat_history: list, use_kb: bool) -> str:
    api_key = os.environ.get("Gemini_Api", "").strip()
    if not api_key:
        return (
            "Gemini API key is not configured. Set the `Gemini_Api` environment variable "
            "and restart Streamlit."
        )

    context_text = (
        retrieve_context(user_message)
        if use_kb
        else "Knowledge-base retrieval is disabled for this conversation."
    )

    try:
        client = get_gemini_client(api_key)
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=make_gemini_contents(user_message, chat_history, context_text),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.35,
                top_p=0.9,
                max_output_tokens=900,
            ),
        )

        answer = (response.text or "").strip()
        if not answer:
            return (
                "I couldn't generate a text response for that message. "
                "Please rephrase your question and try again."
            )
        return answer

    except Exception as exc:
        # Avoid exposing credentials or full request details to end users.
        error_text = str(exc).lower()
        if "api key" in error_text or "permission" in error_text or "unauthenticated" in error_text:
            return (
                "Gemini could not authenticate the request. Check that `Gemini_Api` contains "
                "a valid Gemini API key and that the key has access to the selected model."
            )
        if "quota" in error_text or "resource_exhausted" in error_text or "rate limit" in error_text:
            return (
                "The Gemini API quota or rate limit has been reached. Check your API usage "
                "and try again later."
            )
        if "not found" in error_text or "404" in error_text:
            return (
                f"The selected Gemini model (`{MODEL_NAME}`) was not found or is unavailable "
                "for this API key. Set `GEMINI_MODEL` to a model enabled for your account."
            )
        return (
            "Gemini could not complete the request. Check your internet connection and API "
            "configuration, then try again."
        )


# -------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Preferences")
    st.session_state.user_name = st.text_input(
        "Display name",
        value=st.session_state.user_name,
        max_chars=40,
        help="This name is only used in the chat display.",
    )
    st.session_state.use_knowledge_base = st.toggle(
        "Use public-health CSV knowledge base",
        value=st.session_state.use_knowledge_base,
        help="Pass relevant CSV entries as context to Gemini.",
    )
    st.session_state.show_timestamps = st.toggle(
        "Show message timestamps",
        value=st.session_state.show_timestamps,
    )

    st.divider()
    st.markdown("### Knowledge base")
    st.metric("Available entries", int(len(kb)))
    if kb.empty:
        st.caption("CSV data could not be loaded. Gemini can still answer general questions.")
    else:
        categories = sorted(kb["category"].dropna().unique().tolist())
        st.caption("Loaded: " + ", ".join(categories))

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.caption(f"Gemini model: `{MODEL_NAME}`")
    st.caption("Your API key is read from the environment and is never displayed.")


# -------------------------------------------------------------------
# MAIN HEADER
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {max-width: 860px; padding-top: 2rem; padding-bottom: 3rem;}
    .hero {
        padding: 1.5rem 1.6rem;
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(37,99,235,.09), rgba(16,185,129,.07));
        margin-bottom: 1rem;
    }
    .hero-kicker {
        font-size: .78rem; text-transform: uppercase; letter-spacing: .14em;
        font-weight: 700; opacity: .72; margin-bottom: .45rem;
    }
    .hero h1 {font-size: 2rem; line-height: 1.2; margin: 0 0 .55rem 0;}
    .hero p {font-size: 1rem; opacity: .82; margin: 0; line-height: 1.6;}
    .notice {
        border-left: 3px solid #D97706; padding: .7rem .9rem; margin: 1rem 0;
        background: rgba(217,119,6,.08); border-radius: 0 8px 8px 0;
        font-size: .88rem;
    }
    </style>
    <div class="hero">
        <div class="hero-kicker">Public health · Gemini AI</div>
        <h1>🩺 Crafter's Health Assistant</h1>
        <p>Ask clear questions about general health information, vaccination, and public health.
        Responses may use your project’s CSV knowledge base.</p>
    </div>
    <div class="notice">
        <strong>Important:</strong> This assistant provides general information, not a medical
        diagnosis. For urgent symptoms, contact local emergency services or seek immediate medical care.
    </div>
    """,
    unsafe_allow_html=True,
)

if not os.environ.get("Gemini_Api", "").strip():
    st.warning(
        "Gemini API key not found. Configure the `Gemini_Api` environment variable before sending messages."
    )


# -------------------------------------------------------------------
# CHAT HISTORY RENDER
# -------------------------------------------------------------------
for entry in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(f"**{st.session_state.user_name or 'You'}**")
        st.markdown(entry["user"])
        if st.session_state.show_timestamps:
            st.caption(entry.get("time", ""))

    with st.chat_message("assistant", avatar="🩺"):
        st.markdown(entry["bot"])
        if st.session_state.show_timestamps:
            st.caption(entry.get("time", ""))


# -------------------------------------------------------------------
# CHAT INPUT
# -------------------------------------------------------------------
user_message = st.chat_input("Ask about health, vaccines, or public health…")

if user_message and user_message.strip():
    clean_message = user_message.strip()

    st.session_state.history.append(
        {
            "user": clean_message,
            "bot": "",
            "time": datetime.now(timezone.utc).astimezone().strftime("%d %b %Y, %I:%M %p"),
        }
    )
    current_index = len(st.session_state.history) - 1

    with st.chat_message("user"):
        st.markdown(f"**{st.session_state.user_name or 'You'}**")
        st.markdown(clean_message)

    with st.chat_message("assistant", avatar="🩺"):
        with st.spinner("Thinking with Gemini…"):
            reply = generate_answer(
                clean_message,
                st.session_state.history[:current_index],
                st.session_state.use_knowledge_base,
            )
        st.markdown(reply)

    st.session_state.history[current_index]["bot"] = reply
    st.session_state.history[current_index]["time"] = (
        datetime.now(timezone.utc).astimezone().strftime("%d %b %Y, %I:%M %p")
    )
