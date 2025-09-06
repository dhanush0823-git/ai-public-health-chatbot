# app.py
import streamlit as st
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

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

# ---------------------------
# Built-in disease dictionary
# ---------------------------
DISEASE_KB = {
    "covid": {
        "symptoms": "Fever, dry cough, tiredness, loss of taste or smell.",
        "advice": "Rest, stay hydrated, monitor oxygen levels, isolate if positive.",
        "vaccination": "Ensure you are fully vaccinated and take booster doses if eligible.",
        "doctor_alert": "Consult a doctor if high fever or breathing difficulties occur."
    },
    "dengue": {
        "symptoms": "High fever, severe headache, pain behind eyes, muscle and joint pains, nausea, vomiting.",
        "advice": "Rest, stay hydrated, avoid aspirin, monitor platelet count.",
        "vaccination": "No widely used vaccine for all age groups, consult doctor if travel history present.",
        "doctor_alert": "Seek immediate medical attention if bleeding or severe pain occurs."
    },
    "cold": {
        "symptoms": "Sneezing, runny nose, mild cough, sore throat.",
        "advice": "Rest, drink warm fluids, maintain hygiene.",
        "vaccination": "Flu vaccine recommended annually.",
        "doctor_alert": "Consult doctor if symptoms worsen or persist beyond 7 days."
    },
    # Add more diseases as needed
}

# ---------------------------
# Load FLAN-T5-base model
# ---------------------------
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"  # small CPU-friendly instruct model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    nlp = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=200)
    return nlp

generator = load_model()

# ---------------------------
# LLM RESPONSE FUNCTION
# ---------------------------
def get_llm_response(user_message, context=None):
    # Check disease dictionary first
    lower_msg = user_message.lower()
    for disease, info in DISEASE_KB.items():
        if disease in lower_msg:
            return (
                f"**Symptoms:** {info['symptoms']}\n"
                f"**Advice:** {info['advice']}\n"
                f"**Vaccination tips:** {info['vaccination']}\n"
                f"**Doctor Alert:** {info['doctor_alert']}"
            )
    
    # Otherwise, use the LLM
    prompt = f"""
You are a friendly professional health assistant.
Analyze symptoms, give reassurance for mild issues, preventive advice, vaccination tips, and doctor alerts.
User Name: {st.session_state.user_name}
"""
    if context:
        prompt += f"Previous conversation context: {context}\n"
    prompt += f"User: {user_message}\nAssistant:"

    try:
        response = generator(prompt)
        return response[0]["generated_text"].strip()
    except Exception as e:
        return f"⚠️ Error generating response: {e}"

# ---------------------------
# Sidebar for user info
# ---------------------------
with st.sidebar:
    st.header("👤 Patient Info")
    st.session_state.user_name = st.text_input("Your Name", value=st.session_state.user_name)
    if st.button("🗑 Clear chat"):
        st.session_state.history = []
        st.session_state.disease_context = None
        st.rerun()

# ---------------------------
# Chat Input
# ---------------------------
user_message = st.chat_input("Ask about health, vaccination, or outbreaks...")
if user_message:
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
# Render chat / Welcome
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
        with st.chat_message("user"):
            st.markdown(f"**{entry['name']}:** {entry['user']}")
        with st.chat_message("assistant"):
            st.markdown(entry['bot'])

render_history()
