import os
from datetime import datetime
from io import StringIO

import pandas as pd
import requests
import streamlit as st

from google import genai
from google.genai import types


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Crafter's | AI Health Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "Crafter's Health Assistant"

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-flash-latest"
)


# ============================================================
# 2. GEMINI API KEY
# ============================================================

def get_api_key():
    """
    Read the Gemini API key from Windows environment variables.

    Supports:
        Gemini_api
        Gemini_Api
        GEMINI_API_KEY
    """

    for key_name in (
        "Gemini_api",
        "Gemini_Api",
        "GEMINI_API_KEY",
    ):
        value = os.environ.get(key_name)

        if value and value.strip():
            return value.strip()

    return ""


# ============================================================
# 3. SESSION STATE
# ============================================================

if "user_name" not in st.session_state:
    st.session_state.user_name = "Guest"

if "history" not in st.session_state:
    st.session_state.history = []

if "use_knowledge_base" not in st.session_state:
    st.session_state.use_knowledge_base = True

if "show_timestamps" not in st.session_state:
    st.session_state.show_timestamps = False


# ============================================================
# 4. DATASET URLS
# ============================================================

DATASET_URLS = {
    "Health FAQ": (
        "https://raw.githubusercontent.com/"
        "dhanush0823-git/ai-public-health-chatbot/"
        "main/health_faq.csv"
    ),

    "Vaccination": (
        "https://raw.githubusercontent.com/"
        "dhanush0823-git/ai-public-health-chatbot/"
        "main/vaccination.csv"
    ),

    "Outbreak": (
        "https://raw.githubusercontent.com/"
        "dhanush0823-git/ai-public-health-chatbot/"
        "main/outbreak.csv"
    ),
}


# ============================================================
# 5. LOAD DATASETS
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_dataset(url, category):

    empty_df = pd.DataFrame(
        columns=[
            "question",
            "answer",
            "category",
        ]
    )

    try:
        response = requests.get(
            url,
            timeout=20,
        )

        response.raise_for_status()

        df = pd.read_csv(
            StringIO(
                response.content.decode(
                    "utf-8-sig",
                    errors="replace",
                )
            ),
            on_bad_lines="skip",
        )

        df.columns = [
            str(column).strip().lower()
            for column in df.columns
        ]

        if not {
            "question",
            "answer",
        }.issubset(df.columns):

            return empty_df

        df = df[
            [
                "question",
                "answer",
            ]
        ].copy()

        df["question"] = (
            df["question"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        df["answer"] = (
            df["answer"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        df = df[
            (df["question"] != "")
            & (df["answer"] != "")
        ]

        df["category"] = category

        return df[
            [
                "question",
                "answer",
                "category",
            ]
        ]

    except Exception:
        return empty_df


@st.cache_data(ttl=3600, show_spinner=False)
def load_knowledge_base():

    datasets = []

    for category, url in DATASET_URLS.items():

        df = load_dataset(
            url,
            category,
        )

        if not df.empty:
            datasets.append(df)

    if not datasets:

        return pd.DataFrame(
            columns=[
                "question",
                "answer",
                "category",
            ]
        )

    knowledge_base = pd.concat(
        datasets,
        ignore_index=True,
    )

    knowledge_base = knowledge_base.drop_duplicates(
        subset=[
            "question",
            "answer",
        ]
    )

    return knowledge_base.reset_index(
        drop=True
    )


# Load the datasets.

with st.spinner("Loading health knowledge base..."):

    kb = load_knowledge_base()


# ============================================================
# 6. KNOWLEDGE BASE RETRIEVAL
# ============================================================

def retrieve_context(query, max_items=4):

    if kb.empty:

        return (
            "The CSV knowledge base is unavailable."
        )

    def tokenize(text):

        return {
            word.strip(".,!?;:()[]{}'\"").lower()
            for word in str(text).split()
            if len(
                word.strip(".,!?;:()[]{}'\"")
            ) > 2
        }

    query_terms = tokenize(query)

    if not query_terms:

        return (
            "No matching knowledge-base entries found."
        )

    results = []

    for _, row in kb.iterrows():

        question = str(
            row["question"]
        )

        answer = str(
            row["answer"]
        )

        question_terms = tokenize(
            question
        )

        answer_terms = tokenize(
            answer
        )

        if not question_terms:
            continue

        question_score = (
            len(query_terms & question_terms)
            / len(query_terms)
        )

        answer_score = (
            len(query_terms & answer_terms)
            / len(query_terms)
        )

        score = (
            0.75 * question_score
            + 0.25 * answer_score
        )

        if score > 0:

            results.append(
                (
                    score,
                    question,
                    answer,
                    str(row["category"]),
                )
            )

    results.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    selected = results[:max_items]

    if not selected:

        return (
            "No directly matching knowledge-base "
            "entries were found."
        )

    context_parts = []

    for score, question, answer, category in selected:

        context_parts.append(
            f"""
Category: {category}

Question: {question}

Answer: {answer}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# 7. GEMINI CLIENT
# ============================================================

@st.cache_resource
def get_gemini_client(api_key):

    return genai.Client(
        api_key=api_key
    )


# ============================================================
# 8. MEDICAL SAFETY INSTRUCTIONS
# ============================================================

SYSTEM_INSTRUCTION = """
You are Crafter's Health Assistant, a helpful AI assistant
specializing in general health education and public health.

Your responsibilities include:
- General health information.
- Vaccination education.
- Public health information.
- Explaining health-related concepts.
- Providing information using the supplied knowledge base.

IMPORTANT SAFETY RULES:

1. You are not a doctor.
2. Do not claim to diagnose diseases.
3. Do not invent medical facts, statistics, or sources.
4. Use the supplied knowledge base when it is relevant.
5. If the knowledge base does not support a specific claim,
   clearly communicate uncertainty.
6. Never recommend starting, stopping, or changing
   prescription medication without professional guidance.
7. Encourage users to consult qualified healthcare
   professionals for personal medical decisions.
8. For potentially life-threatening symptoms, such as
   severe breathing difficulty, chest pain, stroke symptoms,
   severe bleeding, or unconsciousness, advise seeking
   emergency medical assistance immediately.
9. Keep answers clear, professional, compassionate,
   and easy to understand.
10. Do not present general information as a personal diagnosis.
"""


# ============================================================
# 9. GEMINI RESPONSE GENERATION
# ============================================================

def generate_answer(
    user_message,
    chat_history,
    use_knowledge_base=True,
):

    api_key = get_api_key()

    # Check the API key.

    if not api_key:

        return (
            "⚠️ Gemini API key not found.\n\n"
            "Please verify that your Windows environment "
            "variable is configured correctly.\n\n"
            "Supported names:\n"
            "- `Gemini_api`\n"
            "- `Gemini_Api`\n"
            "- `GEMINI_API_KEY`\n\n"
            "Restart your terminal and Streamlit after "
            "changing environment variables."
        )

    # Retrieve relevant dataset context.

    if use_knowledge_base:

        knowledge_context = retrieve_context(
            user_message
        )

    else:

        knowledge_context = (
            "Knowledge-base retrieval is disabled."
        )

    # Prepare conversation history.

    contents = []

    for message in chat_history[-8:]:

        previous_user = str(
            message.get("user", "")
        ).strip()

        previous_assistant = str(
            message.get("bot", "")
        ).strip()

        if previous_user:

            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text=previous_user
                        )
                    ],
                )
            )

        if previous_assistant:

            contents.append(
                types.Content(
                    role="model",
                    parts=[
                        types.Part.from_text(
                            text=previous_assistant
                        )
                    ],
                )
            )

    # Construct the latest prompt.

    prompt = f"""
Relevant knowledge-base context:

{knowledge_context}

Latest user question:

{user_message}

Instructions:

Answer the latest question directly.

Use the knowledge-base context when relevant.
Do not assume every retrieved entry is relevant.
If the available information is insufficient,
explain the uncertainty instead of inventing facts.
"""

    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=prompt
                )
            ],
        )
    )

    try:

        client = get_gemini_client(
            api_key
        )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.35,
                top_p=0.9,
                max_output_tokens=900,
            ),
        )

        answer = (
            response.text or ""
        ).strip()

        if not answer:

            return (
                "I couldn't generate a response. "
                "Please rephrase your question and try again."
            )

        return answer

    except Exception as error:

        # Show useful errors without exposing the API key.

        error_message = str(
            error
        ).lower()

        if (
            "api key" in error_message
            or "unauthenticated" in error_message
            or "permission_denied" in error_message
            or "permission denied" in error_message
            or "401" in error_message
            or "403" in error_message
        ):

            return (
                "❌ Gemini authentication failed.\n\n"
                "Verify that your API key is valid and "
                "has access to the selected model."
            )

        if (
            "quota" in error_message
            or "resource_exhausted" in error_message
            or "rate limit" in error_message
            or "429" in error_message
        ):

            return (
                "⚠️ Gemini API quota or rate limit reached. "
                "Check your API usage and try again later."
            )

        if (
            "404" in error_message
            or "not found" in error_message
        ):

            return (
                f"⚠️ Gemini model `{MODEL_NAME}` could not "
                "be found or is unavailable for your API key. "
                "Check the model name and API access."
            )

        # Log the technical details in the terminal.
        # Never print the API key.

        print(
            "Gemini request failed:",
            type(error).__name__,
            str(error)[:500],
        )

        return (
            "❌ Gemini could not complete your request.\n\n"
            "Check your internet connection, API configuration, "
            "and model availability, then try again."
        )


# ============================================================
# 10. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## ⚙️ Preferences"
    )

    st.session_state.user_name = st.text_input(
        "Display name",
        value=st.session_state.user_name,
        max_chars=40,
    )

    st.session_state.use_knowledge_base = st.toggle(
        "Use health CSV knowledge base",
        value=st.session_state.use_knowledge_base,
    )

    st.session_state.show_timestamps = st.toggle(
        "Show timestamps",
        value=st.session_state.show_timestamps,
    )

    st.divider()

    st.markdown(
        "### 📚 Knowledge Base"
    )

    st.metric(
        "Available entries",
        len(kb),
    )

    if not kb.empty:

        categories = sorted(
            kb["category"].unique()
        )

        st.caption(
            "Loaded: " + ", ".join(categories)
        )

    else:

        st.warning(
            "CSV datasets are unavailable."
        )

    st.divider()

    if st.button(
        "🗑 Clear conversation",
        use_container_width=True,
    ):

        st.session_state.history = []

        st.rerun()

    st.divider()

    st.caption(
        f"Gemini model: {MODEL_NAME}"
    )

    st.caption(
        "API credentials are read from environment variables."
    )


# ============================================================
# 11. MAIN HEADER
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 950px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 1.8rem;
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 18px;
        background:
            linear-gradient(
                135deg,
                rgba(37,99,235,.12),
                rgba(16,185,129,.08)
            );
        margin-bottom: 1rem;
    }

    .hero-label {
        font-size: .76rem;
        text-transform: uppercase;
        letter-spacing: .15em;
        font-weight: 700;
        opacity: .7;
    }

    .hero h1 {
        font-size: 2rem;
        margin: .5rem 0;
    }

    .hero p {
        opacity: .8;
        line-height: 1.7;
    }

    .medical-notice {
        padding: .9rem 1rem;
        border-left: 3px solid #D97706;
        background: rgba(217,119,6,.08);
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
    }
    </style>

    <div class="hero">
        <div class="hero-label">
            Public Health · Gemini AI
        </div>

        <h1>🩺 Crafter's Health Assistant</h1>

        <p>
            Your AI-powered assistant for general health education,
            vaccination information, and public health questions.
            Responses can incorporate your project's CSV knowledge base.
        </p>
    </div>

    <div class="medical-notice">
        <strong>Medical disclaimer:</strong>
        This assistant provides general information, not a medical
        diagnosis. For urgent symptoms, seek immediate medical care.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 12. API KEY STATUS
# ============================================================

if not get_api_key():

    st.warning(
        "Gemini API key not found. "
        "Check your Windows environment variable settings "
        "and restart the terminal running Streamlit."
    )


# ============================================================
# 13. DISPLAY CHAT HISTORY
# ============================================================

for entry in st.session_state.history:

    with st.chat_message("user"):

        st.markdown(
            f"**{st.session_state.user_name or 'You'}**"
        )

        st.markdown(
            entry["user"]
        )

        if st.session_state.show_timestamps:

            st.caption(
                entry.get("time", "")
            )

    with st.chat_message(
        "assistant",
        avatar="🩺",
    ):

        st.markdown(
            entry["bot"]
        )

        if st.session_state.show_timestamps:

            st.caption(
                entry.get("time", "")
            )


# ============================================================
# 14. CHAT INPUT
# ============================================================

user_message = st.chat_input(
    "Ask about health, vaccines, or public health..."
)


# ============================================================
# 15. PROCESS USER MESSAGE
# ============================================================

if user_message and user_message.strip():

    user_message = user_message.strip()

    # Preserve the history before adding the new message.

    previous_history = list(
        st.session_state.history
    )

    timestamp = datetime.now().astimezone().strftime(
        "%d %b %Y, %I:%M %p"
    )

    # Display the user's message immediately.

    with st.chat_message("user"):

        st.markdown(
            f"**{st.session_state.user_name or 'You'}**"
        )

        st.markdown(
            user_message
        )

    # Generate the Gemini response.

    with st.chat_message(
        "assistant",
        avatar="🩺",
    ):

        with st.spinner(
            "Thinking with Gemini..."
        ):

            bot_reply = generate_answer(
                user_message=user_message,
                chat_history=previous_history,
                use_knowledge_base=(
                    st.session_state.use_knowledge_base
                ),
            )

        st.markdown(
            bot_reply
        )

    # Save the conversation.

    st.session_state.history.append(
        {
            "user": user_message,
            "bot": bot_reply,
            "time": timestamp,
        }
    )
