# AI-Powered Statistics Learning Assistant

## Project Info
| | |
|---|---|
| **Domain** | Data Science / Statistics Education |
| **Creator** | *Mattia Zen* |
| **Model** | Gemini 2.5 Flash via LangChain |

---

## What It Does
This is personalised AI tutoring assistant that helps users learn data science and statistics concepts. It adapts its explanations, examples, and depth based on the user's level, learning goal, and background domain. It supports multi-turn conversation and will support document-based Q&A (RAG) in a future release.


### Data Flow
1. User submits a question along with their profile (level, goal, background)
2. `build_system_prompt()` constructs a personalised system prompt injecting the user profile
3. `build_user_prompt()` composes the user turn with chain-of-thought and format directives
4. The LangChain chain sends the full conversation history + composed prompts to the Gemini API
5. The answer is returned and the conversation history is updated for the next turn
6. *(Future)* Retrieved document chunks from the RAG pipeline will be injected at step 3

---

## Project Structure

```
LLM_app/
│
├── backend/                  # Core application logic
│   ├── __init__.py
│   ├── config.py             # Centralised settings (model, temperature, paths)
│   ├── llm_client.py         # LLM API calls and conversation management
│   ├── prompts.py            # System prompt, user prompt, and RAG context builders
│   └── rag_pipeline.py       # Document ingestion and retrieval (in progress)
│
├── prompts/
│   └── system_prompt.txt     # Base system prompt loaded at runtime
│
├── tests/
│   ├── __init__.py
│   └── test_llm_client.py    # Unit tests (mocked, no API calls)
│
├── .env                      # API keys — never committed (see setup below)
├── .gitignore
├── main.py                   # Entry point for local testing
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create and activate a virtual environment
```bash
# Mac / Linux
python -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your API key
Create a `.env` file in the project root (never commit this file):
```
GOOGLE_API_KEY=your-gemini-api-key-here
```
Get your key from [Google AI Studio](https://aistudio.google.com/).

### 5. Add the system prompt
Create `prompts/system_prompt.txt` and write your base system prompt inside it. Example:
```
You are a helpful and patient data science tutor.
Your goal is to help students understand statistical and machine learning concepts clearly.
Always be encouraging and precise.
```

### 6. Run locally
```bash
python main.py
```

### 7. Run tests
```bash
pytest tests/test_llm_client.py -v
```

