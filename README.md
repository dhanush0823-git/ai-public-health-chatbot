🩺 AI-Driven Public Health Chatbot for Disease Awareness

A **multilingual Streamlit-based chatbot** that helps users with:

* 💡 General health FAQs (symptoms, prevention, treatment)
* 💉 Vaccination schedules and boosters
* 🚨 Outbreak alerts and health advisories

Designed to support **rural and semi-urban communities**, the chatbot loads datasets directly from GitHub and provides an easy-to-use web interface.
🔹 Features

* Multilingual support (English, Hindi, Tamil, Telugu, Marathi, Bengali)
* Simple and interactive Streamlit UI
* Datasets: health FAQs, vaccination info, outbreak alerts
* Cloud-ready, can be scaled for WhatsApp/SMS integration
🔹 Project Structure
ai-public-health-chatbot/
│
├── app.py                 # Streamlit chatbot code
├── requirements.txt       # dependencies
├── data/
│   ├── health_faq.csv     # general disease FAQs
│   ├── vaccination.csv    # vaccination schedules
│   └── outbreak.csv       # outbreak alerts
🔹 Installation & Run
# Install dependencies
pip install -r requirements.txt

# Run chatbot
streamlit run app.py
🔹 Future Scope

* Integration with WhatsApp/SMS for wider reach
* Voice input for non-typing users
* Real-time government health database integration
* Analytics dashboard for awareness tracking
