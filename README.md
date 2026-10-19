<div align="center">

# 🩺 AI Health Assistant

### AI-powered health education, vaccination awareness, and public-health information.

A health-focused conversational assistant built with **Streamlit**, **Google Gemini**, and a curated CSV knowledge base.

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-Application-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Google%20Gemini-Generative%20AI-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white" alt="Google Gemini">
  <img src="https://img.shields.io/badge/Domain-Healthcare-159D80?style=for-the-badge" alt="Healthcare">
</p>

**[Overview](#-overview) · [Features](#-key-features) · [Architecture](#-how-it-works) · [Setup](#-getting-started) · [Deployment](#-deploy-on-streamlit-community-cloud)**

</div>

---

## 📌 Overview

**AI Health Assistant** is a health-domain conversational application designed to help users explore general health information, disease prevention, vaccination, wellness, and public-health topics through a chat interface.

The application uses **Google Gemini** to classify incoming questions and generate responses. It can also retrieve potentially relevant entries from three CSV datasets and provide them to Gemini as supporting context.

The assistant is intentionally scoped to health-related conversations. If a question is unrelated to health—such as general celebrity or sports trivia—it returns a polite message explaining the assistant's purpose rather than answering the unrelated question.

> **Important:** This project provides general educational information. It is not a medical diagnostic system and does not replace a qualified healthcare professional.

## ✨ Key Features

- **Gemini-powered responses** — generates conversational answers using Google's Gemini API.
- **Health-focused scope control** — classifies the user's question before generating a full answer and declines unrelated requests.
- **CSV knowledge base** — loads Health FAQ, Vaccination, and Outbreak datasets from GitHub.
- **Context-assisted answers** — selects candidate dataset entries using lightweight word-overlap retrieval and passes them to Gemini when relevant.
- **Conversation context** — includes a limited window of previous messages to support follow-up questions.
- **Premium Streamlit interface** — dark visual theme, subtle gradients, responsive layout, chat bubbles, and suggested starter questions.
- **Configurable knowledge base** — users can enable or disable dataset context from the sidebar.
- **Dataset refresh control** — clears Streamlit's dataset cache and reloads the CSV sources.
- **API-key protection** — reads the Gemini key from Streamlit Secrets or environment variables instead of hardcoding it.
- **Model discovery** — attempts to list models available to the configured API key; an optional `GEMINI_MODEL` setting can pin a preferred model.
- **Helpful error handling** — displays more actionable messages for missing credentials, authentication failures, unavailable models, and API quota issues.
- **Medical safety guidance** — includes general safeguards and emergency-care guidance for potentially life-threatening symptoms.

## 🖥️ Application Preview

The interface includes:
- A branded **AI Health Assistant** header.
- Suggested questions to help users start a conversation.
- A chat area for user prompts and assistant responses.
- A sidebar for display preferences, knowledge-base status, API configuration status, and conversation controls.

_Add a screenshot of your deployed app to the repository (for example, `assets/app-preview.png`) and uncomment the image below if you want it displayed here._

<!--
<p align="center">
  <img src="assets/app-preview.png" alt="AI Health Assistant application preview" width="900">
</p>
-->

## 🧰 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| Streamlit | Web interface and session state |
| Google Gen AI SDK (`google-genai`) | Gemini model access and text generation |
| Google Gemini | Health-topic classification and response generation |
| Pandas | CSV loading and data processing |
| Requests | Downloading datasets from GitHub |
| Streamlit caching | Caching dataset results and the Gemini client |

## 📂 Project Structure

```text
ai-public-health-chatbot/
├── app.py              # Streamlit UI, dataset retrieval, Gemini integration
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
├── health_faq.csv      # Health FAQ dataset (GitHub source)
├── vaccination.csv     # Vaccination dataset (GitHub source)
└── outbreak.csv        # Public-health outbreak dataset (GitHub source)
```

The application currently loads the three CSV files from their raw GitHub URLs. The CSV files do not have to be stored beside `app.py` for the deployed application to retrieve them.

## 🔍 How It Works

```text
                    ┌────────────────────────┐
                    │      User question     │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ Streamlit chat UI      │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ Read Gemini API key    │
                    │ from Secrets/env       │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ Gemini scope classifier│
                    └────────────┬───────────┘
                                 │
                      ┌──────────┴──────────┐
                      │                     │
                      ▼                     ▼
              Health-related           Out of scope
                      │                     │
                      ▼                     ▼
          ┌────────────────────┐   ┌────────────────────┐
          │ Retrieve candidate │   │ Return the health- │
          │ CSV context        │   │ scope message      │
          └─────────┬──────────┘   └────────────────────┘
                    │
                    ▼
          ┌────────────────────┐
          │ Combine context,   │
          │ history, and query │
          └─────────┬──────────┘
                    │
                    ▼
          ┌────────────────────┐
          │ Gemini generates   │
          │ health response    │
          └─────────┬──────────┘
                    │
                    ▼
          ┌────────────────────┐
          │ Display and store  │
          │ response in chat   │
          └────────────────────┘
```

### Request flow

1. The user submits a question in the Streamlit chat.
2. The app reads the Gemini API key from Streamlit Secrets or the process environment.
3. Gemini classifies the request as health-related or out of scope.
4. If the request is out of scope, the app returns a fixed scope message.
5. If the request is health-related, the app optionally retrieves candidate entries from the CSV knowledge base.
6. The app combines relevant context, recent conversation turns, and the user's latest question.
7. Gemini generates the response, which is displayed in the chat and retained in the current Streamlit session.

**Implementation note:** The current retrieval method is lightweight lexical matching based on word overlap. It is not vector search and may miss relevant entries when a user phrases a question differently from the dataset.

## 📚 Knowledge Base

The project uses three public CSV sources hosted in this GitHub repository:

| Dataset | Intended use |
|---|---|
| `health_faq.csv` | General health questions and answers |
| `vaccination.csv` | Vaccination-related information |
| `outbreak.csv` | Public-health and outbreak information |

The loader expects each CSV to contain columns named `question` and `answer` (column names are normalized to lowercase). Rows with missing or blank questions or answers are discarded. The app adds a `category` field to identify the source dataset.

Dataset retrieval is optional. If a CSV cannot be loaded, Gemini may still respond to an in-scope question using its general knowledge, while communicating uncertainty where appropriate.

> Dataset content and public-health information can become outdated. Review and update the CSV data before relying on it for current guidance.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or newer is recommended.
- A Google Gemini API key.
- Git, if cloning the repository.
- Internet access to retrieve the CSV files and call the Gemini API.

### 1. Clone the repository

```bash
git clone https://github.com/dhanush0823-git/ai-public-health-chatbot.git
cd ai-public-health-chatbot
```

### 2. Create and activate a virtual environment (recommended)

**Windows — Command Prompt**

```cmd
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` should include the packages used by the app:

```text
streamlit
pandas
requests
google-genai
```

### 4. Configure your Gemini API key

The application checks these names:

- `Gemini_api`
- `Gemini_Api`
- `GEMINI_API_KEY`

#### Option A — Windows Command Prompt (temporary)

```cmd
set Gemini_api=YOUR_GEMINI_API_KEY
```

Then, in the same terminal, run the app:

```cmd
streamlit run app.py
```

This `set` command applies only to the current Command Prompt session.

#### Option B — macOS / Linux (temporary)

```bash
export Gemini_api="YOUR_GEMINI_API_KEY"
streamlit run app.py
```

#### Option C — Streamlit Community Cloud (recommended for deployment)

1. Push the application code to GitHub.
2. Open the app in Streamlit Community Cloud.
3. Open the app's **Settings → Secrets**.
4. Add the following TOML entry:

```toml
Gemini_api = "YOUR_GEMINI_API_KEY"
```

5. Save the configuration and reboot the app if needed.

**Never commit your actual API key to GitHub.** Do not paste it into source code, screenshots, issue reports, or public logs.

### 5. Run the application locally

```bash
streamlit run app.py
```

Streamlit will provide a local URL in the terminal, typically `http://localhost:8501`.

## ☁️ Deploy on Streamlit Community Cloud

1. Ensure `app.py`, `requirements.txt`, and `README.md` are in the root of your GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new app and select the GitHub repository and branch.
4. Set the main file path to `app.py`.
5. Deploy the app.
6. Open **Settings → Secrets** and add `Gemini_api` using the TOML format shown above.
7. Reboot the app if the new secret is not detected.
8. Test one health-related prompt and one unrelated prompt.

Keep the repository public only if you are comfortable sharing the source code and CSV data. Keep API credentials in Streamlit Secrets.

## ⚙️ Configuration

| Setting | Default / behavior | Description |
|---|---|---|
| `Gemini_api` | Read from Streamlit Secrets or environment | Preferred API-key name |
| `Gemini_Api` | Fallback key name | Alternative spelling supported by the app |
| `GEMINI_API_KEY` | Fallback key name | Common environment variable alternative |
| `GEMINI_MODEL` | Unset by default | Optional model ID to prefer; otherwise the app attempts model discovery |
| Knowledge-base toggle | Enabled | Controls whether candidate CSV context is sent to Gemini |
| Timestamp toggle | Disabled | Shows timestamps for messages |
| Clear conversation | Button | Clears chat history in the current Streamlit session |
| Refresh CSV datasets | Button | Clears the cached CSV data and reloads the sources |

Model availability depends on the Google AI project, API key, SDK version, and current API offerings. If model discovery fails, inspect the Streamlit app logs and confirm that the Gemini API is enabled and the key has access to a text-generation model.

## 🛡️ Safety, Privacy, and Limitations

- **Not a diagnostic tool:** The assistant provides general educational information, not a medical diagnosis or personalized treatment plan.
- **Seek professional care:** Consult a qualified healthcare professional for individual medical decisions.
- **Emergency situations:** If symptoms may be life-threatening, seek emergency medical assistance immediately.
- **AI can be wrong:** Gemini may produce incomplete, outdated, or inaccurate information. Verify important claims with reliable clinical or public-health sources.
- **Scope classification is probabilistic:** The classifier can misclassify ambiguous prompts. It is an application guardrail, not a guarantee.
- **Knowledge-base matching is lexical:** Word-overlap retrieval is simple and may not find semantically similar questions. It can also return weakly related entries; Gemini is instructed to use context only when relevant.
- **Dataset freshness:** The CSV sources need ongoing review and maintenance.
- **Session history:** Conversation history is kept in Streamlit session state by this app; this implementation does not provide a persistent user database or long-term account history.
- **API usage:** The app makes a Gemini call to classify a question and, for in-scope questions, another call to generate an answer. This affects latency and API usage.
- **Secrets:** Never expose the API key in source code or public repositories.

This project is intended for learning, demonstration, and general health education. It has not been presented as a clinically validated medical device.

## 🧪 Example Questions

### Health-related questions

- What are common ways to reduce the spread of infections?
- Why are vaccinations important for public health?
- What is hypertension?
- What are common symptoms of dehydration?
- How do public-health teams respond to disease outbreaks?

### Out-of-scope question

**Prompt:** `Who is Virat Kohli?`

**Expected behavior:** The assistant should explain that it is designed for health-related questions and invite the user to ask about health, diseases, symptoms, prevention, vaccination, or public health.

## 🧯 Troubleshooting

### “Gemini API key not found”

- On Streamlit Community Cloud, verify the `Gemini_api` entry under **Settings → Secrets**.
- Ensure the value is quoted correctly in TOML.
- Save the Secrets configuration and reboot the app.
- Confirm the secret is configured for the correct Streamlit app.

### “Gemini authentication failed”

- Check that the key is valid.
- Confirm that the Gemini API is enabled for the associated Google project.
- Ensure the key has not been revoked or restricted in a way that blocks the request.
- Never share the key in a public issue or screenshot.

### “No available Gemini model” or model not found

- Check which models are available to your API key and project.
- Update the `google-genai` package if needed.
- Set `GEMINI_MODEL` in Streamlit Secrets to an available model ID if automatic discovery does not work.
- Review the application logs for the underlying error.

### CSV datasets are unavailable

- Check that the GitHub repository and CSV files are accessible.
- Verify that each CSV has `question` and `answer` columns.
- Use **Refresh CSV datasets** in the sidebar after correcting the source files.
- The app may still answer in-scope questions with Gemini when CSV data is unavailable.

### API quota or rate limit reached

Check your Gemini API usage and limits. The app uses a separate classification request before generating an in-scope answer.

## 🗺️ Possible Future Improvements

- Replace lexical retrieval with sentence embeddings and semantic search.
- Add a vector database for scalable retrieval-augmented generation (RAG).
- Add dataset source attribution and display which entries informed an answer.
- Add automated tests for dataset loading, scope classification, and API error handling.
- Add an evaluation set for health relevance, retrieval quality, and response safety.
- Add stronger monitoring for stale or conflicting public-health entries.
- Add optional multilingual support and accessible UI refinements.
- Add persistent chat history only with appropriate consent, security, and retention controls.

## 🤝 Contributing

Contributions and suggestions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make a focused change.
4. Test the application locally.
5. Open a pull request describing the change and its impact.

Please do not include API keys, private user data, or unverified medical claims in contributions.

## 📄 License

No license is specified in this README. Unless a license file is added to the repository, users should not assume that the source code is available for unrestricted reuse. Add a `LICENSE` file if you want to grant others explicit reuse rights.

## 👨‍💻 Author

**Dhanush S**

- GitHub: [@dhanush0823-git](https://github.com/dhanush0823-git)
- Project repository: [AI Health Assistant](https://github.com/dhanush0823-git/ai-public-health-chatbot)

---

<div align="center">

**AI Health Assistant**  
*Making general health information easier to explore—one question at a time.*

</div>
