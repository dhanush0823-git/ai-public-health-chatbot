import os
import re
from datetime import datetime
from io import StringIO
from typing import Any

import pandas as pd
import requests
import streamlit as st
from google import genai
from google.genai import types


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Health Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "Health Assistant"
# Optionally set GEMINI_MODEL in Streamlit Secrets to pin a specific model.
# Otherwise, discover a text-generation model available to this API key.
MODEL_NAME = os.getenv("GEMINI_MODEL", "").strip()


DATASET_URLS = {
    "Health FAQ": (
        "https://raw.githubusercontent.com/"
        "dhanush0823-git/ai-public-health-chatbot/main/health_faq.csv"
    ),
    "Vaccination": (
        "https://raw.githubusercontent.com/"
        "dhanush0823-git/ai-public-health-chatbot/main/vaccination.csv"
    ),
    "Outbreak": (
        "https://raw.githubusercontent.com/"
        "dhanush0823-git/ai-public-health-chatbot/main/outbreak.csv"
    ),
}

SYSTEM_INSTRUCTION = """
You are Health Assistant, an AI assistant dedicated ONLY to health and healthcare topics.

IN-SCOPE topics include diseases, symptoms, medical conditions, health education,
wellness, nutrition when health-related, mental health, first aid, prevention,
vaccination, public health, medical tests, healthcare systems, and health-related
questions about medicines.

OUT-OF-SCOPE RULE — MUST FOLLOW:
If the user's request is not meaningfully related to health, disease, healthcare,
medicine, wellness, or public health, do NOT answer the actual question, do NOT
provide facts about the unrelated topic, and do NOT continue the unrelated discussion.
Reply with this message, exactly:
"I'm Health Assistant, designed to help with health, diseases, symptoms, prevention, vaccination, and public health. Your question isn't related to my purpose, but I'm happy to help with a health-related question."

If the request is ambiguous, ask one short clarification to determine whether it is health-related.
A person's name alone does not make a question health-related. For example, "Who is Virat Kohli?"
is out of scope and must receive the exact out-of-scope message above.

MEDICAL SAFETY:
- Do not claim to diagnose or replace a qualified healthcare professional.
- Do not invent medical facts, statistics, sources, outbreak updates, or vaccine schedules.
- Clearly communicate uncertainty when evidence or context is insufficient.
- Do not recommend starting, stopping, or changing prescription medicines.
- Encourage professional care for individual medical decisions.
- For potentially life-threatening symptoms (severe breathing difficulty, chest pain,
  stroke symptoms, unconsciousness, or severe bleeding), advise immediate emergency care.
- Keep in-scope answers clear, compassionate, and reasonably concise.
"""


# ============================================================
# GLOBAL STYLE
# ============================================================
st.markdown(
    """
    <style>
    :root {
        --accent: #78DCC0;
        --accent-strong: #2CBF9F;
        --panel: rgba(255,255,255,.035);
        --border: rgba(255,255,255,.10);
        --muted: #A6B0BF;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, rgba(44,191,159,.09), transparent 32%),
            radial-gradient(ellipse at 90% 8%, rgba(91,117,255,.08), transparent 28%);
    }

    .block-container {
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 6rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1.2rem;
    }

    .brand-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 48px;
        height: 48px;
        border-radius: 15px;
        background: linear-gradient(135deg, rgba(44,191,159,.22), rgba(91,117,255,.18));
        border: 1px solid rgba(120,220,192,.28);
        font-size: 1.55rem;
    }

    .brand-name {
        font-size: 1.05rem;
        font-weight: 750;
        line-height: 1.2;
        letter-spacing: -.02em;
    }

    .brand-caption {
        color: var(--muted);
        font-size: .78rem;
        margin-top: 4px;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: clamp(1.5rem, 4vw, 2.8rem);
        border-radius: 26px;
        border: 1px solid var(--border);
        background: linear-gradient(120deg, rgba(21,35,46,.94), rgba(18,32,34,.84));
        margin-bottom: 1.15rem;
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 250px;
        height: 250px;
        right: -95px;
        top: -120px;
        border-radius: 50%;
        background: rgba(44,191,159,.12);
        filter: blur(5px);
        pointer-events: none;
    }

    .eyebrow {
        color: var(--accent);
        font-size: .74rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: .16em;
        margin-bottom: .8rem;
    }

    .hero h1 {
        font-size: clamp(2rem, 4.4vw, 3.1rem);
        line-height: 1.08;
        letter-spacing: -.045em;
        margin: 0 0 .9rem 0;
        font-weight: 800;
    }

    .hero p {
        color: #C4CED8;
        max-width: 680px;
        line-height: 1.75;
        font-size: 1.02rem;
        margin: 0;
    }

    .hero-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 1.3rem;
    }

    .hero-tag {
        border: 1px solid rgba(255,255,255,.12);
        background: rgba(255,255,255,.045);
        color: #D7E1E8;
        border-radius: 999px;
        padding: 6px 11px;
        font-size: .78rem;
    }

    .notice {
        border: 1px solid rgba(242,181,85,.20);
        border-left: 3px solid #E8B45A;
        border-radius: 12px;
        background: rgba(232,180,90,.065);
        padding: .85rem 1rem;
        color: #E8DFCF;
        font-size: .86rem;
        line-height: 1.55;
        margin-bottom: 1.5rem;
    }

    .section-label {
        color: var(--muted);
        text-transform: uppercase;
        font-size: .73rem;
        letter-spacing: .14em;
        font-weight: 750;
        margin: 1.5rem 0 .8rem 0;
    }

    .suggestion {
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1rem;
        min-height: 105px;
        background: var(--panel);
    }

    .suggestion-title {
        font-weight: 700;
        margin-bottom: .35rem;
    }

    .suggestion-copy {
        color: var(--muted);
        font-size: .86rem;
        line-height: 1.5;
    }

    .status-card {
        border: 1px solid var(--border);
        background: var(--panel);
        border-radius: 15px;
        padding: .9rem 1rem;
        margin: .7rem 0;
    }

    .status-title {
        font-weight: 700;
        font-size: .9rem;
        margin-bottom: .25rem;
    }

    .status-copy {
        color: var(--muted);
        font-size: .82rem;
        line-height: 1.5;
    }

    div[data-testid="stChatMessage"] {
        border: 1px solid rgba(255,255,255,.055);
        border-radius: 18px;
        padding: .85rem 1rem;
        margin-bottom: .75rem;
    }

    div[data-testid="stChatInput"] {
        border-radius: 18px;
    }

    .stButton > button {
        border-radius: 12px;
        transition: all .18s ease;
    }

    .stButton > button:hover {
        border-color: var(--accent);
        transform: translateY(-1px);
    }

    .footer-note {
        color: var(--muted);
        font-size: .78rem;
        text-align: center;
        padding: 1.5rem .5rem .4rem .5rem;
        line-height: 1.6;
    }

    @media (max-width: 700px) {
        .block-container { padding-top: 1rem; }
        .hero { border-radius: 20px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================
def init_state() -> None:
    defaults = {
        "user_name": "Dhanush",
        "history": [],
        "use_knowledge_base": True,
        "show_timestamps": False,
        "pending_prompt": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# ============================================================
# API KEY: STREAMLIT CLOUD SECRETS FIRST, THEN ENVIRONMENT
# ============================================================
def get_api_key() -> str:
    names = ("Gemini_api", "Gemini_Api", "GEMINI_API_KEY")

    # Streamlit Community Cloud Secrets
    for name in names:
        try:
            value = st.secrets.get(name, "")
            if value and str(value).strip():
                return str(value).strip()
        except Exception:
            # st.secrets can raise when no secrets file exists.
            pass

    # Local environment variables
    for name in names:
        value = os.environ.get(name, "")
        if value and value.strip():
            return value.strip()

    return ""


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_dataset(url: str, category: str) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["question", "answer", "category"])

    try:
        response = requests.get(url, timeout=(5, 20))
        response.raise_for_status()
        df = pd.read_csv(
            StringIO(response.content.decode("utf-8-sig", errors="replace")),
            on_bad_lines="skip",
        )
    except Exception:
        return empty

    df.columns = [str(c).strip().lower() for c in df.columns]

    if not {"question", "answer"}.issubset(df.columns):
        return empty

    df = df[["question", "answer"]].copy()
    df["question"] = df["question"].fillna("").astype(str).str.strip()
    df["answer"] = df["answer"].fillna("").astype(str).str.strip()
    df = df[(df["question"] != "") & (df["answer"] != "")]
    df["category"] = category
    return df[["question", "answer", "category"]]


@st.cache_data(ttl=3600, show_spinner="Loading knowledge base...")
def load_knowledge_base() -> pd.DataFrame:
    frames = [
        load_dataset(url, category)
        for category, url in DATASET_URLS.items()
    ]
    frames = [frame for frame in frames if not frame.empty]

    if not frames:
        return pd.DataFrame(columns=["question", "answer", "category"])

    result = pd.concat(frames, ignore_index=True)
    return result.drop_duplicates(
        subset=["question", "answer"]
    ).reset_index(drop=True)


kb = load_knowledge_base()


# ============================================================
# RETRIEVAL
# ============================================================
def tokenize(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+", str(text))
        if len(token) > 2
    }


def retrieve_context(query: str, max_items: int = 4) -> str:
    if kb.empty:
        return "The CSV knowledge base is currently unavailable."

    query_terms = tokenize(query)
    if not query_terms:
        return "No matching knowledge-base entries were found."

    ranked: list[tuple[float, str, str, str]] = []

    for _, row in kb.iterrows():
        question = str(row["question"])
        answer = str(row["answer"])
        q_terms = tokenize(question)
        a_terms = tokenize(answer)

        if not q_terms:
            continue

        q_score = len(query_terms & q_terms) / len(query_terms)
        a_score = len(query_terms & a_terms) / len(query_terms)
        score = 0.75 * q_score + 0.25 * a_score

        if score > 0:
            ranked.append(
                (score, question, answer, str(row["category"]))
            )

    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = ranked[:max_items]

    if not selected:
        return "No directly matching knowledge-base entries were found."

    return "\n\n".join(
        f"[{category}]\nQuestion: {question}\nAnswer: {answer}"
        for _, question, answer, category in selected
    )


# ============================================================
# GEMINI
# ============================================================
@st.cache_resource
def get_gemini_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def available_gemini_models(client: genai.Client) -> list[str]:
    """Discover models available to this API key instead of guessing model IDs."""
    candidates = []

    if MODEL_NAME:
        candidates.append(MODEL_NAME)

    try:
        for model in client.models.list():
            name = str(getattr(model, "name", "") or "").strip()
            actions = getattr(model, "supported_actions", None) or []
            actions = [str(action).lower() for action in actions]

            if name.startswith("models/"):
                name = name.split("/", 1)[1]

            lower_name = name.lower()
            if not lower_name.startswith("gemini"):
                continue
            if any(term in lower_name for term in ("embedding", "aqa")):
                continue
            if actions and not any("generatecontent" in action for action in actions):
                continue

            candidates.append(name)

    except Exception as exc:
        print(
            f"Could not list Gemini models ({type(exc).__name__}): "
            f"{str(exc)[:250]}"
        )

    models = list(dict.fromkeys(candidates))

    if not MODEL_NAME:
        models.sort(
            key=lambda name: (
                "flash" not in name.lower(),
                "pro" in name.lower(),
                name.lower(),
            )
        )

    return models


def call_gemini(
    client: genai.Client,
    contents: list,
    system_instruction: str,
    temperature: float = 0.1,
) -> str:
    models = available_gemini_models(client)

    if not models:
        raise RuntimeError(
            "No Gemini text-generation model was listed for this API key. "
            "Check API access or set GEMINI_MODEL to a model available to this project."
        )

    last_error = None

    for model_name in models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temperature,
                    top_p=0.9,
                    max_output_tokens=1000,
                ),
            )

            answer = (response.text or "").strip()
            if answer:
                return answer

        except Exception as exc:
            last_error = exc
            message = str(exc).lower()

            # Retry another model only when this model is unavailable.
            if any(term in message for term in (
                "404", "not found", "not supported", "unknown model",
                "model is not available", "unsupported model"
            )):
                continue
            raise

    if last_error:
        raise last_error

    raise RuntimeError("Gemini returned an empty response for available models.")


OUT_OF_SCOPE_REPLY = (
    "I'm Health Assistant, designed to help with health, diseases, symptoms, "
    "prevention, vaccination, and public health. Your question isn't related "
    "to my purpose, but I'm happy to help with a health-related question."
)

SCOPE_CLASSIFIER_INSTRUCTION = """
You are a strict topic classifier for a health-only assistant.
Classify the latest user message as HEALTH_RELATED or OUT_OF_SCOPE.

HEALTH_RELATED includes medical conditions, symptoms, wellness, nutrition for health,
mental health, medicine, first aid, vaccination, disease prevention, public health,
and healthcare services.

OUT_OF_SCOPE includes sports, celebrities, entertainment, general trivia, coding,
math, politics, travel, and topics not meaningfully related to health.

Classify the user's actual intent. A person's name or a health-related word mentioned
in passing does not automatically make the request health-related.
Return exactly one word: HEALTH_RELATED or OUT_OF_SCOPE.
"""


def is_health_related(client: genai.Client, user_message: str) -> bool:
    """Classify the request before generating an answer."""
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=f"Classify this user message:\n{user_message}"
                )
            ],
        )
    ]

    result = call_gemini(
        client=client,
        contents=contents,
        system_instruction=SCOPE_CLASSIFIER_INSTRUCTION,
        temperature=0,
    )

    # Fail closed if classification is unclear.
    return result.strip().upper().startswith("HEALTH_RELATED")


def generate_answer(
    user_message: str,
    history: list[dict[str, Any]],
    use_knowledge_base: bool,
) -> str:
    api_key = get_api_key()

    if not api_key:
        return (
            "I couldn't find the Gemini API key in this deployed app. "
            "Open your Streamlit app's **Settings → Secrets**, verify that "
            "`Gemini_api` is present and saved, then reboot the app."
        )

    try:
        client = get_gemini_client(api_key)

        # Enforce the health-only boundary before generating an answer.
        if not is_health_related(client, user_message):
            return OUT_OF_SCOPE_REPLY

        context = (
            retrieve_context(user_message)
            if use_knowledge_base
            else "Knowledge-base retrieval is disabled by the user."
        )

        contents = []

        # Add recent conversation turns for continuity.
        for turn in history[-8:]:
            previous_user = str(turn.get("user", "")).strip()
            previous_bot = str(turn.get("bot", "")).strip()

            if previous_user and previous_bot:
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=previous_user)],
                    )
                )
                contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=previous_bot)],
                    )
                )

        prompt = f"""
Relevant project knowledge-base context:
{context}

Latest user question:
{user_message}

Answer the latest question only if it is health-related.
Use the supplied context only when relevant. It may be incomplete or irrelevant.
If it does not answer the question, use general health knowledge carefully and
communicate uncertainty. Do not force irrelevant CSV entries into the answer.
"""

        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        )

        return call_gemini(
            client=client,
            contents=contents,
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.25,
        )

    except Exception as exc:
        message = str(exc).lower()
        print(
            f"Gemini request failed ({type(exc).__name__}): "
            f"{str(exc)[:400]}"
        )

        if any(term in message for term in (
            "api key", "unauthenticated", "permission_denied",
            "permission denied", "401", "403"
        )):
            return (
                "Gemini authentication failed. Verify that the key in "
                "Streamlit Secrets is valid and that the Gemini API is enabled."
            )

        if any(term in message for term in (
            "quota", "resource_exhausted", "rate limit", "429"
        )):
            return (
                "The Gemini API quota or rate limit has been reached. "
                "Check your API usage and try again later."
            )

        if any(term in message for term in (
            "404", "not found", "not supported", "unknown model",
            "no gemini text-generation model"
        )):
            return (
                "I couldn't find an available Gemini text-generation model for this API key. "
                "Check the models enabled for your Google AI project, or set `GEMINI_MODEL` "
                "in Streamlit Secrets to a model ID available to your project."
            )

        return (
            "Gemini couldn't complete this request. Check the Streamlit app logs "
            "for technical details, then verify model availability and API access."
        )


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div class="brand-row">
            <div class="brand-icon">🩺</div>
            <div>
                <div class="brand-name">Health Assistant</div>
                <div class="brand-caption">AI-powered health guidance</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Preferences")
    st.session_state.user_name = st.text_input(
        "Display name",
        value=st.session_state.user_name,
        max_chars=40,
    )

    st.session_state.use_knowledge_base = st.toggle(
        "Use project knowledge base",
        value=st.session_state.use_knowledge_base,
        help="Pass matching CSV entries to Gemini as additional context.",
    )

    st.session_state.show_timestamps = st.toggle(
        "Show timestamps",
        value=st.session_state.show_timestamps,
    )

    st.divider()
    st.markdown("### Knowledge base")
    st.metric("Available entries", len(kb))

    if not kb.empty:
        categories = sorted(kb["category"].dropna().unique().tolist())
        st.caption("Loaded: " + " · ".join(categories))
    else:
        st.caption("CSV data is unavailable. Gemini can still be used.")

    st.divider()
    api_key_found = bool(get_api_key())

    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-title">
                {"🟢 Gemini configuration detected" if api_key_found else "🟠 Gemini key not detected"}
            </div>
            <div class="status-copy">
                {"The app can read a key from Secrets or environment variables." if api_key_found else "Check Settings → Secrets and confirm the key name is Gemini_api."}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        f"Preferred model: `{MODEL_NAME}`"
        if MODEL_NAME
        else "Model: automatically selected from those available to your API key"
    )

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.pending_prompt = ""
        st.rerun()

    if st.button("Refresh CSV datasets", use_container_width=True):
        load_knowledge_base.clear()
        load_dataset.clear()
        st.rerun()


# ============================================================
# MAIN HERO
# ============================================================
st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">YOUR HEALTH, EXPLAINED · POWERED BY GEMINI</div>
        <h1>Your health questions,<br>answered with clarity.</h1>
        <p>
            Ask questions about general health, vaccination, and public health.
            Get conversational explanations powered by Gemini, with relevant
            information from your project's CSV knowledge base when available.
        </p>
        <div class="hero-tags">
            <span class="hero-tag">✦ Gemini AI</span>
            <span class="hero-tag">▤ Project knowledge base</span>
            <span class="hero-tag">◉ Conversational answers</span>
            <span class="hero-tag">✓ Health-focused responses</span>
        </div>
    </section>
    <div class="notice">
        <strong>Health information notice</strong> &nbsp; This assistant provides
        general educational information, not diagnosis or treatment. For personal
        medical decisions, consult a qualified healthcare professional. If you
        may be experiencing an emergency, seek emergency medical care immediately.
    </div>
    """,
    unsafe_allow_html=True,
)

# Friendly first-run view and prompt suggestions.
if not st.session_state.history:
    st.markdown('<div class="section-label">Start a conversation</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    suggestions = [
        (
            "💉 Vaccination",
            "Why are vaccinations important for public health?",
        ),
        (
            "🛡️ Prevention",
            "What are practical ways to prevent the spread of infections?",
        ),
        (
            "🌍 Public health",
            "How do public health teams respond to disease outbreaks?",
        ),
    ]

    for column, (title, prompt_text) in zip((col1, col2, col3), suggestions):
        with column:
            st.markdown(
                f"""
                <div class="suggestion">
                    <div class="suggestion-title">{title}</div>
                    <div class="suggestion-copy">Explore this topic with the assistant.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                "Ask this question",
                key=f"suggest_{title}",
                use_container_width=True,
            ):
                st.session_state.pending_prompt = prompt_text
                st.rerun()


# ============================================================
# CHAT HISTORY
# ============================================================
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


# ============================================================
# CHAT INPUT AND RESPONSE
# ============================================================
typed_message = st.chat_input(
    "Ask about health, vaccines, or public health…"
)

user_message = typed_message or st.session_state.pending_prompt

if user_message and user_message.strip():
    st.session_state.pending_prompt = ""
    user_message = user_message.strip()

    prior_history = list(st.session_state.history)
    timestamp = datetime.now().astimezone().strftime("%d %b %Y · %I:%M %p")

    with st.chat_message("user"):
        st.markdown(f"**{st.session_state.user_name or 'You'}**")
        st.markdown(user_message)

    with st.chat_message("assistant", avatar="🩺"):
        with st.spinner("Thinking through your question…"):
            reply = generate_answer(
                user_message=user_message,
                history=prior_history,
                use_knowledge_base=st.session_state.use_knowledge_base,
            )
        st.markdown(reply)

    st.session_state.history.append(
        {
            "user": user_message,
            "bot": reply,
            "time": timestamp,
        }
    )

st.markdown(
    """
    <div class="footer-note">
        Health Assistant · General health education only<br>
        AI-generated information can be incomplete or incorrect. Verify important
        medical information with a qualified professional.
    </div>
    """,
    unsafe_allow_html=True,
)
