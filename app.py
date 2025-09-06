# app.py
import streamlit as st
from datetime import datetime
from typing import List, Dict, Any

# Transformers for LLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

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
    st.session_state.history = []  # list of dicts {"user":..., "bot":..., "time":...}

if "disease_detected" not in st.session_state:
    st.session_state.disease_detected = None

# ---------------------------
# DISEASE KNOWLEDGE BASE
# ---------------------------
DISEASE_KB = {
    "covid": {
        "symptoms": "Fever, dry cough, tiredness, loss of taste or smell.",
        "advice": "Rest, stay hydrated, isolate if positive, monitor oxygen levels.",
        "vaccination": "Ensure full vaccination and take booster doses if eligible.",
        "doctor_alert": "Consult a doctor if high fever or breathing difficulties occur."
    },
    "dengue": {
        "symptoms": "High fever, severe headache, pain behind eyes, joint/muscle pain, rash.",
        "advice": "Stay hydrated, rest, avoid mosquito bites.",
        "vaccination": "No widely used vaccine; avoid mosquito exposure.",
        "doctor_alert": "Seek medical attention if fever is very high or bleeding occurs."
    },
    "cold": {
        "symptoms": "Sneezing, runny nose, sore throat, mild cough.",
        "advice": "Rest, hydrate, take warm fluids.",
        "vaccination": "No specific vaccine; practice hygiene and flu shots if seasonal.",
        "doctor_alert": "See doctor if fever is persistent or symptoms worsen."
    }
}

# ---------------------------
# LLM MODEL LOAD
# ---------------------------
@st.cache_resource
def load_model():
    model_name = "lmsys/vicuna-13b-delta-v1.1"  # replace with medical LLaMA URL if needed
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype="auto"
    )
    chatbot_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512
    )
    return chatbot_pipeline

chatbot = load_model()

# ---------------------------
# LLM RESPONSE FUNCTION
# ---------------------------
def get_llm_response(user_message: str, context: List[Dict[str, str]] = None) -> str:
    # Handle fear/anxiety keywords separately
    fear_keywords = ["die", "afraid", "panic", "worried", "scared"]
    if any(word in user_message.lower() for word in fear_keywords):
        return ("I understand your concern. Most illnesses are manageable. "
                "Please monitor your symptoms, rest, stay hydrated, and consult a doctor if severe. "
                "You are not alone, and help is available.")

    # Check disease KB
    for disease, info in DISEASE_KB.items():
        if disease in user_message.lower():
            st.session_state.disease_detected = disease
            bot_answer = (
                f"**Symptoms:** {info['symptoms']}\n"
                f"**Advice:** {info['advice']}\n"
                f"**Vaccination tips:** {info['vaccination']}\n"
                f"**Doctor Alert:** {info['doctor_alert']}"
            )
            if context and bot_answer in [b['bot'] for b in context]:
                bot_answer += "\n(Please consult a doctor if symptoms persist.)"
            return bot_answer

    # Build conversation context
    structured_context = ""
    if context:
        structured_context = "\n".join([f"User: {entry['user']}\nAssistant: {entry['bot']}" for entry in context[-5:]])

    prompt = f"""
You are a professional and friendly health assistant.
- Analyze symptoms, provide advice, vaccination tips, doctor alerts.
- Avoid repeating previous responses.
- Be concise, empathetic, and safe.
- Use previous 5 turns as context if available.

Conversation context:
{structured_context}

User: {user_message}
Assistant:"""

    try:
        response = chatbot(prompt, max_length=200, do_sample=True, temperature=0.7)
        bot_answer = response[0]["generated_text"].strip()
        # Avoid exact repetition
        recent_answers = [entry['bot'] for entry in context] if context else []
        if bot_answer in recent_answers:
            bot_answer += " (Please follow up with a doctor if symptoms change or persist.)"
        return bot_answer
    except Exception as e:
        return f"⚠️ Error generating response: {e}"

# ---------------------------
# STREAMLIT SIDEBAR
# ---------------------------
with st.sidebar:
    st.header("👤 Patient Info")
    st.session_state.user_name = st.text_input("Your Name", value=st.session_state.user_name)
    if st.button("🗑 Clear chat"):
        st.session_state.history = []
        st.session_state.disease_detected = None
        st.experimental_rerun()

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.markdown(f"<h1 style='text-align:center'>{APP_TITLE}</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Chat input
user_message = st.chat_input("Ask about health, vaccination, or outbreaks...")

if user_message:
    bot_reply = get_llm_response(user_message, st.session_state.history)
    st.session_state.history.append({
        "user": user_message,
        "bot": bot_reply,
        "time": datetime.utcnow().isoformat()
    })

# Display chat history
for entry in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(f"**{st.session_state.user_name}:** {entry['user']}")
    with st.chat_message("assistant"):
        st.markdown(entry['bot'])

# ---------------------------
# Dashboard: Detected Disease
# ---------------------------
if st.session_state.disease_detected:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("🩺 Detected Disease")
    disease = st.session_state.disease_detected
    info = DISEASE_KB[disease]
    st.markdown(f"**Disease:** {disease.capitalize()}")
    st.markdown(f"**Symptoms:** {info['symptoms']}")
    st.markdown(f"**Advice:** {info['advice']}")
    st.markdown(f"**Vaccination Tips:** {info['vaccination']}")
    st.markdown(f"**Doctor Alert:** {info['doctor_alert']}")
