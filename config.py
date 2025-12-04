# backend/config.py
"""
Central configuration for ITP ConversX
All API keys, URLs, and settings in one place
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ==================== OPENAI CONFIG ====================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "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")
OPENAI_PROJECT_ID = os.environ.get("OPENAI_PROJECT_ID", "proj_mA8J7MQh8U7rDgM7PtEBZZAi")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
GPT_MODEL = "gpt-3.5-turbo"
TTS_MODEL = "tts-1"
DEFAULT_VOICE = "alloy"

# ==================== QDRANT CONFIG ====================
QDRANT_URL = os.environ.get("QDRANT_URL", "https://8c72d910-3e8c-4176-a12e-19ad5f0ff42d.eu-central-1-0.aws.cloud.qdrant.io")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.R7kK9WQkQz9dCt7eWbv0PScTU_TsDXIRN_ZcO1Eg4j0")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "itp-embeddings")
VECTOR_SIZE = int(os.environ.get("VECTOR_SIZE", "384"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "100"))
QDRANT_DUPLICATE_SCORE_THRESHOLD = float(os.environ.get("QDRANT_DUP_SCORE", "0.90"))

# ==================== OLLAMA CONFIG ====================
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = "mistral:7b-instruct"

# ==================== DIRECTORIES ====================
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", os.path.join(os.getcwd(), "uploads"))
CHUNKS_DIR = os.environ.get("CHUNKS_DIR", os.path.join(os.getcwd(), "chunks"))

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)

# ==================== ITP PROMPT ====================
ITP_SYSTEM_PROMPT = (
    "You are ITP ConversX, the Islamabad Traffic Police AI Assistant for officers and public users. "
    "Use the provided CONTEXT (extracted from official ITP documents) and answer the QUESTION concisely, "
    "factually, and in a helpful officer-style tone. Keep responses short (1-2 sentences). "
    "Do not invent laws. If context doesn't contain the answer, say you don't have enough information and suggest "
    "which document or office to consult."
)

# ==================== EXPORTS ====================
__all__ = [
    'OPENAI_API_KEY',
    'OPENAI_PROJECT_ID',
    'OPENAI_API_URL',
    'GPT_MODEL',
    'TTS_MODEL',
    'DEFAULT_VOICE',
    'QDRANT_URL',
    'QDRANT_API_KEY',
    'COLLECTION',
    'VECTOR_SIZE',
    'BATCH_SIZE',
    'QDRANT_DUPLICATE_SCORE_THRESHOLD',
    'OLLAMA_API_URL',
    'OLLAMA_MODEL',
    'UPLOAD_DIR',
    'CHUNKS_DIR',
    'ITP_SYSTEM_PROMPT'
]

