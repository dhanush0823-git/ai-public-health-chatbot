import streamlit as st
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import pandas as pd

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

if "disease_detected" not in st.session_state:
    st.session_state.disease_detected = None

# ---------------------------
# Disease Dictionary
# ---------------------------
DISEASE_KB = {
    "covid": {
        "symptoms": "Fever, dry cough, tiredness, loss of taste or smell.",
        "advice": "Rest, stay hydrated, monitor oxygen levels, isolate if positive.",
        "vaccination": "Ensure you are fully vaccinated and take booster doses if eligible.",
        "doctor_alert": "Consult a doctor if high fever or breathing difficulties occur.",
        "severity": "Moderate"
    },
    "dengue": {
        "symptoms": "High fever, severe headache, pain behind eyes, muscle and joint pains, nausea, vomiting.",
        "advice": "Rest, stay hydrated, avoid aspirin, monitor platelet count.",
        "vaccination": "No widely used vaccine for all age groups, consult doctor if travel history present.",
        "doctor_alert": "Seek immediate medical attention if bleeding or severe pain occurs.",
        "severity": "Moderate"
    },
    "cold": {
        "symptoms": "Sneezing, runny nose, mild cough, sore throat.",
        "advice": "Rest, drink warm fluids, maintain hygiene.",
        "vaccination": "Flu vaccine recommended annually.",
        "doctor_alert": "Consult doctor if symptoms worsen or persist beyond 7 days.",
        "severity": "Mild"
    }
}

# ---------------------------
# Load FLAN-T5-base model
# ---------------------------
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    nlp = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=200)
    return nlp

generator = load_model()

# ---------------------------
# LLM RESPONSE FUNCTION
# ---------------------------
def get_llm_response(user_message, context=None):
    lower_msg = user_message.lower()
    # Check disease dictionary
    for disease, info in DISEASE_KB.items():
        if disease in lower_msg:
            st.session_state.disease_detected = disease
            return (
                f"**Symptoms:** {info['symptoms']}\n"
                f"**Advice:** {info['advice']}\n"
                f"**Vaccination tips:** {info['vaccination']}\n"
                f"**Doctor Alert:** {info['doctor_alert']}"
            )
    # Unknown / general questions
    prompt = f"""
You are a friendly professional health assistant.
Explain clearly and briefly for general health questions.
Avoid repeating the user's name unnecessarily.

Previous conversation context: {context if context else 'None'}
User question: {user_message}
Assistant:
"""
    try:
        response = generator(prompt)
        return response[0]["generated_text"].strip()
    except Exception as e:
        return f"⚠️ Error generating response: {e}"

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("👤 Patient Info")
    st.session_state.user_name = st.text_input("Your Name", value=st.session_state.user_name)
    if st.button("🗑 Clear chat"):
        st.session_state.history = []
        st.session_state.disease_detected = None
        st.rerun()

# ---------------------------
# Chat Input
# ---------------------------
user_message = st.chat_input("Ask about health, vaccination, or outbreaks...")
if user_message:
    context_text = "\n".join([f"{e['user']}: {e['bot']}" for e in st.session_state.history[-5:]])
    answer_text = get_llm_response(user_message, context=context_text)
    
    st.session_state.history.append({
        "user": user_message,
        "bot": answer_text,
        "name": st.session_state.user_name,
        "time": datetime.utcnow().isoformat()
    })

# ---------------------------
# Render Chat (ChatGPT style)
# ---------------------------
for entry in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(f"**{entry['name']}:** {entry['user']}")
    with st.chat_message("assistant"):
        st.markdown(entry['bot'])

# ---------------------------
# Health Dashboard
# ---------------------------
if st.session_state.disease_detected:
    disease = st.session_state.disease_detected
    info = DISEASE_KB[disease]
    st.markdown("---")
    st.markdown(f"### 🩺 Health Dashboard: {disease.capitalize()}")
    df = pd.DataFrame({
        "Field": ["Symptoms", "Advice", "Vaccination tips", "Doctor Alert", "Severity"],
        "Information": [info["symptoms"], info["advice"], info["vaccination"], info["doctor_alert"], info["severity"]]
    })
    st.table(df)
