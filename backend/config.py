import os

# Model settings
GEMINI_MODEL = "gemini-2.5-flash"
TEMPERATURE = 0.3
MAX_TOKENS = 1500

# API key — loaded from environment 
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")

# RAG settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 3

# Path to system prompt file (relative to project root)
SYSTEM_PROMPT_PATH = "prompts/system_prompt.txt"
