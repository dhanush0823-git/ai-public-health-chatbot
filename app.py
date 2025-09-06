# app.py
import streamlit as st
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import pandas as pd

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="Crafter's", layout="wide")
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
if "welcome_shown" not in st.session_state:
    st.session_state.welcome_shown = False

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
        "vaccination": "No widely used vaccine for all age groups; consult doctor if travel history present.",
        "doctor_alert": "Seek immediate medical attention if bleeding or severe pain occurs.",
        "severity": "Moderate"
    },
    "cold": {
        "symptoms": "Sneezing, runny nose, mild cough, sore throat.",
        "advice": "Rest, drink warm fluids, maintain hygiene.",
        "vaccination": "Flu vaccine recommended annually.",
        "doctor_alert": "Consult doctor if symptoms worsen or persist beyond 7 days.",
        "severity": "Mild"
    },
    "malaria": {
        "symptoms": "Fever, chills, sweating, headache, nausea, vomiting.",
        "advice": "Rest, stay hydrated, take prescribed antimalarial drugs.",
        "vaccination": "No widely available vaccine; consult doctor for prophylaxis.",
        "doctor_alert": "Seek immediate medical attention if fever is high or persistent.",
        "severity": "Moderate"
    },
    "fever": {
        "symptoms": "High temperature, sweating, fatigue.",
        "advice": "Rest, stay hydrated, monitor temperature.",
        "vaccination": "Depends on cause; consult doctor if unsure.",
        "doctor_alert": "See a doctor if fever persists >3 days or is very high.",
        "severity": "Mild"
    },
    "headache": {
        "symptoms": "Pain in head or temples, sensitivity to light, fatigue.",
        "advice": "Rest, stay hydrated, take OTC pain relievers if needed.",
        "vaccination": "Not applicable.",
        "doctor_alert": "See a doctor if headaches are severe or recurrent.",
        "severity": "Mild"
    }
}

# ---------------------------
# Load LLM model
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

    # Anxiety/fear handling
    fear_keywords = ["die", "afraid", "panic", "worried", "scared"]
    if any(word in lower_msg for word in fear_keywords):
        return ("I understand your concern. Most illnesses are manageable. "
                "Please monitor your symptoms, rest, stay hydrated, and consult a doctor if severe. "
                "You are not alone, and help is available.")

    # Known disease
    for disease, info in DISEASE_KB.items():
        if disease in lower_msg:
            st.session_state.disease_detected = disease
            bot_answer = (
                f"**Symptoms:** {info['symptoms']}\n"
                f"**Advice:** {info['advice']}\n"
                f"**Vaccination tips:** {info['vaccination']}\n"
                f"**Doctor Alert:** {info['doctor_alert']}"
            )
            # Avoid repeated answer
            if context and bot_answer in [b for u, b in context]:
                bot_answer += "\n(Additional note: consult your doctor if symptoms persist.)"
            return bot_answer

    # Build structured context
    structured_context = ""
    if context:
        structured_context = "\n".join([f"User: {u}\nAssistant: {b}" for u, b in context[-5:]])

    # Prompt
    prompt = f"""
You are a professional, friendly health assistant.
- Analyze symptoms, give preventive advice, vaccination tips, doctor alerts.
- Avoid repeating sentences or answers.
- Do not copy previous replies exactly; vary phrasing.
- Use last 5 conversation turns as context for follow-ups.
- Provide realistic advice.

Previous conversation:
{structured_context}

User input: {user_message}
Assistant:
"""
    try:
        response = generator(prompt)
        bot_answer = response[0]["generated_text"].strip()
        # Filter exact repetition
        recent_answers = [b for u, b in context] if context else []
        if bot_answer in recent_answers:
            bot_answer += " (Please follow up if symptoms change or persist.)"
        return bot_answer
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
        st.session_state.welcome_shown = False
        st.rerun()

# ---------------------------
# Main UI
# ---------------------------
st.markdown(f"<h1 style='text-align:center'>{APP_TITLE}</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Welcome
if not st.session_state.welcome_shown:
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
    st.session_state.welcome_shown = True

# ---------------------------
# Chat input
# ---------------------------
user_message = st.chat_input("Ask about health, vaccination, or outbreaks...")
if user_message:
    context_text = [(e['user'], e['bot']) for e in st.session_state.history[-5:]]
    answer_text = get_llm_response(user_message, context=context_text)
    st.session_state.history.append({
        "user": user_message,
        "bot": answer_text,
        "name": st.session_state.user_name,
        "time": datetime.utcnow().isoformat()
    })

# ---------------------------
# Render Chat
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
    
    severity_color = {"Mild": "green", "Moderate": "orange", "Severe": "red"}
    
    dashboard_df = pd.DataFrame({
        "Field": ["Symptoms", "Advice", "Vaccination tips", "Doctor Alert", "Severity"],
        "Information": [info["symptoms"], info["advice"], info["vaccination"], info["doctor_alert"], info["severity"]]
    })
    
    st.table(dashboard_df)
    st.markdown(
        f"<p style='font-weight:bold'>Severity: <span style='color:{severity_color.get(info['severity'],'black')}'>{info['severity']}</span></p>",
        unsafe_allow_html=True
    )
