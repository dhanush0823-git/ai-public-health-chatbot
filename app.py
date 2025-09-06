# app.py
import streamlit as st
from datetime import datetime
import os

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="Crafter's", layout="centered")
APP_TITLE = "🩺 Crafter's — AI Health Chatbot"

# ---------------------------
# SESSION STATE INIT
# ---------------------------
if "user_name" not in st.session_state:
    st.session_state.user_name = "Guest"

if "history" not in st.session_state:
    st.session_state.history = []

if "disease_context" not in st.session_state:
    st.session_state.disease_context = None

if "api_key" not in st.session_state:
    # Use environment variable as default, can be updated in sidebar
    st.session_state.api_key = os.getenv("OPENAI_API_KEY", "")

# ---------------------------
# OPENAI IMPORT
# ---------------------------
try:
    import openai
except ImportError:
    st.error("The 'openai' package is not installed. Run 'pip install openai'.")
    st.stop()

# ---------------------------
# Sidebar for API key and user info
# ---------------------------
with st.sidebar:
    st.header("👤 Patient Info / API Key")
    st.session_state.user_name = st.text_input("Your Name", value=st.session_state.user_name)
    
    if not st.session_state.api_key:
        st.session_state.api_key = st.text_input(
            "Enter OpenAI API Key (saved for session)", 
            type="password"
        )

    if st.button("🗑 Clear chat"):
        st.session_state.history = []
        st.session_state.disease_context = None
        st.rerun()

# Validate API key
if not st.session_state.api_key:
    st.warning("⚠️ OpenAI API key not set. Enter your key in the sidebar to continue.")
    st.stop()

# Initialize OpenAI client
client = openai.OpenAI(api_key=st.session_state.api_key)

# ---------------------------
# LLM RESPONSE FUNCTION (New API)
# ---------------------------
def get_llm_response(user_message, context=None):
    prompt = f"""
You are a professional and friendly health assistant.
Analyze symptoms, give reassurance for mild issues, preventive advice, vaccination tips, and doctor alerts.
User Name: {st.session_state.user_name}
"""
    if context:
        prompt += f"Previous conversation context: {context}\n"
    prompt += f"User: {user_message}\nAssistant:"

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error generating response: {e}"

# ---------------------------
# Chat Input
# ---------------------------
user_message = None
try:
    user_message = st.chat_input("Ask about health, vaccination, or outbreaks...")
except Exception:
    user_message = st.text_input("Ask about health, vaccination, or outbreaks:")

if user_message:
    # Collect last 5 messages as context
    context_text = "\n".join([f"{e['user']}: {e['bot']}" for e in st.session_state.history[-5:]])
    answer_text = get_llm_response(user_message, context=context_text)
    
    entry = {
        "user": user_message,
        "bot": answer_text,
        "name": st.session_state.user_name,
        "time": datetime.utcnow().isoformat()
    }
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
                <p>I’m <b>Crafter's</b>, your AI health assistant. I can help with:</p>
                <ul style='text-align:left; max-width:500px; margin:auto;'>
                    <li>🩺 Symptom analysis & advice</li>
                    <li>💉 Vaccination guidance</li>
                    <li>🚨 Outbreak alerts</li>
                    <li>❓ General health FAQs</li>
                </ul>
                <p>Type your question below to start the conversation.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    welcome_displayed = False
    for entry in st.session_state.history:
        if not welcome_displayed:
            st.markdown(f"👋 Welcome back, **{st.session_state.user_name}**!")
            welcome_displayed = True
        try:
            with st.chat_message("user"):
                st.markdown(f"**{entry['name']}:** {entry['user']}")
        except Exception:
            st.markdown(f"**{entry['name']}:** {entry['user']}")
        try:
            with st.chat_message("assistant"):
                st.markdown(entry['bot'])
        except Exception:
            st.markdown(entry['bot'])

render_history()
