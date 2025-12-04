# rag_backend_itp.py
"""
Multi-City ITP ConversX backend:
- Supports Islamabad, Karachi, and Multan Traffic Police
- City-specific collections in Qdrant
- City-aware prompts for Mistral and OpenAI
- RAG pipeline with intelligent chunking and deduplication
"""

import os
import re
import uuid
import json
import logging
from typing import List

import jwt
import time




LIVEKIT_URL = os.getenv("LIVEKIT_URL", "wss://livechatbot-o9mjs452.livekit.cloud")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "APIv6DxoJniCz7V")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "L3ZJhLIJnDxKXF824aWcbQPYKAQWV1VIXbOhIvqo4XA")

import requests
import pandas as pd
import nltk
import numpy as np

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from pypdf import PdfReader

nltk.download("punkt", quiet=True)
logging.basicConfig(level=logging.INFO)

# ------------------- CONFIG -------------------
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", os.path.join(os.getcwd(), "uploads"))
CHUNKS_DIR = os.environ.get("CHUNKS_DIR", os.path.join(os.getcwd(), "chunks"))
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)

QDRANT_URL = os.environ.get("QDRANT_URL", "https://8c72d910-3e8c-4176-a12e-19ad5f0ff42d.eu-central-1-0.aws.cloud.qdrant.io")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.R7kK9WQkQz9dCt7eWbv0PScTU_TsDXIRN_ZcO1Eg4j0")
VECTOR_SIZE = int(os.environ.get("VECTOR_SIZE", "384"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "100"))
QDRANT_DUPLICATE_SCORE_THRESHOLD = float(os.environ.get("QDRANT_DUP_SCORE", "0.90"))

OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api/chat")
OPENAI_API_URL = os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "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")
OPENAI_PROJECT_ID = "proj_mA8J7MQh8U7rDgM7PtEBZZAi"

# City-specific configurations
CITY_CONFIGS = {
    "islamabad": {
        "name": "Islamabad Traffic Police",
        "short_name": "ITP",
        "collection": "itp-embeddings",
        "system_prompt": (
            "You are ITP ConversX, the Islamabad Traffic Police AI Assistant for officers and public users. "
            "Use the provided CONTEXT (extracted from official ITP documents) and answer the QUESTION concisely, "
            "factually, and in a helpful officer-style tone. Keep responses short (1-2 sentences). "
            "Do not invent laws. If context doesn't contain the answer, say you don't have enough information and suggest "
            "which document or office to consult. Always identify yourself as Islamabad Traffic Police."
        ),
        "ollama_model": "islamabad-traffic:latest"
        
    },
    "karachi": {
        "name": "Karachi Traffic Police",
        "short_name": "KTP",
        "collection": "ktp-embeddings",
        "system_prompt": (
            "You are KTP ConversX, the Karachi Traffic Police AI Assistant for officers and public users. "
            "Use the provided CONTEXT (extracted from official KTP documents) and answer the QUESTION concisely, "
            "factually, and in a helpful officer-style tone. Keep responses short (1-2 sentences). "
            "Do not invent laws. If context doesn't contain the answer, say you don't have enough information and suggest "
            "which document or office to consult. Always identify yourself as Karachi Traffic Police."
        ),
        "ollama_model": "karachi-traffic:latest"
    },
    "multan": {
        "name": "Multan Traffic Police",
        "short_name": "MTP",
        "collection": "mtp-embeddings",
        "system_prompt": (
            "You are MTP ConversX, the Multan Traffic Police AI Assistant for officers and public users. "
            "Use the provided CONTEXT (extracted from official MTP documents) and answer the QUESTION concisely, "
            "factually, and in a helpful officer-style tone. Keep responses short (1-2 sentences). "
            "Do not invent laws. If context doesn't contain the answer, say you don't have enough information and suggest "
            "which document or office to consult. Always identify yourself as Multan Traffic Police."
        ),
        "ollama_model": "multan-traffic:latest"
    }
}

# ------------------- INIT MODELS & CLIENTS -------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

qdrant_kwargs = {"url": QDRANT_URL}
if QDRANT_API_KEY:
    qdrant_kwargs["api_key"] = QDRANT_API_KEY
client = QdrantClient(**qdrant_kwargs)

# Create collections for each city
for city, config in CITY_CONFIGS.items():
    try:
        if not client.collection_exists(config["collection"]):
            client.create_collection(
                collection_name=config["collection"],
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            logging.info(f"Created Qdrant collection: {config['collection']}")
    except Exception as e:
        logging.warning(f"Collection check/create error for {city}: {e}")

# TTS/STT imports
TTS = None
STT = None
try:
    from services.tts_service import text_to_speech as tts_text_to_speech
    TTS = tts_text_to_speech
    logging.info("Imported TTS service")
except Exception:
    logging.info("TTS module not available")

try:
    from services.stt_service import speech_to_text as stt_speech_to_text
    STT = stt_speech_to_text
    logging.info("Imported STT service")
except Exception:
    logging.info("STT module not available")

# ------------------- FASTAPI APP -------------------
app = FastAPI(title="Multi-City ITP ConversX")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def index():
    return {
        "status": "Multi-City ITP ConversX backend running",
        "cities": list(CITY_CONFIGS.keys())
    }

def translate_to_urdu(text: str) -> str:
    """Uses OpenAI for translation."""
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "Translate the following English text into formal Urdu. Keep tone polite and clear."},
                {"role": "user", "content": text},
            ],
        }
        resp = requests.post(OPENAI_API_URL, headers=headers, json=data, timeout=10)
        resp.raise_for_status()
        translated = resp.json()["choices"][0]["message"]["content"].strip()
        return translated
    except Exception as e:
        logging.warning(f"Translation to Urdu failed: {e}")
        return text

# ------------------- UTIL FUNCTIONS -------------------
def is_page_type_page(text: str) -> bool:
    keywords = ["appendix", "index", "references", "bibliography"]
    return any(kw in text.lower() for kw in keywords)

def clean_pdf(file_path: str) -> str:
    """Read PDF pages; merge consecutive short pages; skip index/appendix pages."""
    reader = PdfReader(file_path)
    pages = list(reader.pages)
    filtered_pages = []
    prev_short = None
    for i, page in enumerate(pages):
        page_text = page.extract_text() or ""
        page_text = page_text.strip()
        if is_page_type_page(page_text):
            continue
        if len(page_text.split()) < 10:
            if prev_short is None:
                prev_short = page_text
            else:
                prev_short += " " + page_text
            continue
        if prev_short:
            page_text = prev_short + " " + page_text
            prev_short = None
        filtered_pages.append(page_text)
    return "\n\n".join(filtered_pages)

def sentence_chunk(text: str, max_sentences: int = 5, overlap: int = 2) -> List[str]:
    """Chunk text by sentences with overlap."""
    sents = nltk.sent_tokenize(text)
    chunks = []
    i = 0
    while i < len(sents):
        chunk = " ".join(sents[i : i + max_sentences]).strip()
        if len(chunk.split()) >= 5 and not re.fullmatch(r"[\d\s\W]+", chunk):
            chunks.append(chunk)
        i += max_sentences - overlap
    return chunks

def is_valid_chunk(chunk: str, min_words: int = 5, min_chars: int = 30) -> bool:
    chunk = chunk.strip()
    if len(chunk.split()) < min_words or len(chunk) < min_chars:
        return False
    if not any(c.isalpha() for c in chunk):
        return False
    if re.fullmatch(r"[-_=*#\s]+", chunk):
        return False
    words = chunk.split()
    non_alpha_ratio = sum(1 for w in words if not any(c.isalpha() for c in w)) / max(len(words), 1)
    return non_alpha_ratio <= 0.8

def deduplicate_embeddings(embeddings, texts, threshold: float = 0.90):
    """Remove near-duplicate texts based on cosine similarity."""
    from sklearn.metrics.pairwise import cosine_similarity
    keep = [True] * len(texts)
    for i in range(len(embeddings)):
        if not keep[i]:
            continue
        sims = cosine_similarity([embeddings[i]], embeddings[i + 1 :])[0] if i + 1 < len(embeddings) else []
        for j, sim in enumerate(sims, start=i + 1):
            if sim > threshold:
                keep[j] = False
    filtered_texts = [t for k, t in zip(keep, texts) if k]
    filtered_embs = [e for k, e in zip(keep, embeddings) if k]
    return filtered_texts, filtered_embs

def filter_against_qdrant(collection_name: str, texts: List[str], embeddings, score_threshold: float = QDRANT_DUPLICATE_SCORE_THRESHOLD):
    """Check against Qdrant to filter duplicates."""
    unique_texts = []
    unique_embeddings = []
    for txt, emb in zip(texts, embeddings):
        try:
            res = client.search(collection_name=collection_name, query_vector=emb.tolist(), limit=1, with_payload=False)
            if not res or res[0].score < score_threshold:
                unique_texts.append(txt)
                unique_embeddings.append(emb)
            else:
                logging.debug(f"Skipped duplicate chunk in {collection_name} with score {res[0].score:.3f}")
        except Exception as e:
            logging.warning(f"Qdrant search failed: {e}; keeping chunk")
            unique_texts.append(txt)
            unique_embeddings.append(emb)
    return unique_texts, unique_embeddings

# ------------------- LLM FUNCTIONS -------------------
def ask_mistral(context: str, question: str, city: str = "islamabad") -> str:
    """Call Ollama Mistral with city-specific model."""
    config = CITY_CONFIGS.get(city, CITY_CONFIGS["islamabad"])
    system_prompt = config["system_prompt"]
    model = config.get("ollama_model", "mistral:7b-instruct")
    
    prompt = f"{system_prompt}\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}\n"
    data = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    
    try:
        headers = {"Content-Type": "application/json"}
        resp = requests.post(OLLAMA_API_URL, json=data, headers=headers, timeout=15, stream=True)
        resp.raise_for_status()
        answer = ""
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                obj = json.loads(line)
                content = obj.get("message", {}).get("content", "")
                if content:
                    answer += content
            except Exception:
                continue
        return answer.strip()
    except Exception as e:
        logging.warning(f"Ollama/Mistral call failed for {city}: {e}")
        return ""

def ask_openai(context: str, question: str, city: str = "islamabad") -> str:
    """Fallback to OpenAI with city-specific prompt."""
    if not OPENAI_API_KEY:
        return ""
    
    config = CITY_CONFIGS.get(city, CITY_CONFIGS["islamabad"])
    system_prompt = config["system_prompt"]
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    if OPENAI_PROJECT_ID:
        headers["OpenAI-Project"] = OPENAI_PROJECT_ID
    
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
        ],
        "temperature": 0.0,
        "max_tokens": 256,
    }
    
    try:
        resp = requests.post(OPENAI_API_URL, headers=headers, json=data, timeout=15)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logging.warning(f"OpenAI call failed for {city}: {e}")
        return ""

# ------------------- API ENDPOINTS -------------------
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), city: str = Form("islamabad")):
    """Upload and process document for specific city."""
    if city not in CITY_CONFIGS:
        return JSONResponse(status_code=400, content={"error": f"Invalid city: {city}"})
    
    config = CITY_CONFIGS[city]
    collection_name = config["collection"]
    
    # Ensure collection exists
    try:
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
    except Exception as e:
        logging.warning(f"Collection init error for {city}: {e}")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    ext = file.filename.lower().split(".")[-1]
    logging.info(f"Received upload for {city}: {file.filename} (ext={ext})")
    
    text = ""
    if ext == "pdf":
        text = clean_pdf(file_path)
    elif ext == "txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as r:
            text = r.read()
    elif ext == "csv":
        try:
            df = pd.read_csv(file_path, dtype=str, encoding="utf-8", on_bad_lines="skip")
            text = "\n".join(df.fillna("").astype(str).values.flatten())
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": "CSV read failed"})
    elif ext in ("xls", "xlsx"):
        try:
            df = pd.read_excel(file_path, dtype=str)
            text = "\n".join(df.fillna("").astype(str).values.flatten())
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": "Excel read failed"})
    else:
        return JSONResponse(status_code=400, content={"error": "Unsupported file type"})
    
    if not text or len(text.strip()) < 50:
        return {"status": "ok", "chunks": 0, "message": "No meaningful text extracted."}
    
    # Chunk and validate
    raw_chunks = sentence_chunk(text, max_sentences=3, overlap=1)
    valid_chunks = [c for c in raw_chunks if is_valid_chunk(c)]
    
    if not valid_chunks:
        return {"status": "ok", "chunks": 0, "message": "No valid chunks after filtering."}
    
    # Embed
    try:
        embeddings = embedder.encode(valid_chunks, convert_to_numpy=True, show_progress_bar=True)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Embedding failed"})
    
    # Deduplicate
    dedup_chunks, dedup_embeddings = deduplicate_embeddings(embeddings, valid_chunks)
    unique_chunks, unique_embeddings = filter_against_qdrant(collection_name, dedup_chunks, dedup_embeddings)
    
    if not unique_chunks:
        return {"status": "ok", "chunks": 0, "message": "No new content (all duplicates)."}
    
    # Prepare points
    ids = [str(uuid.uuid4()) for _ in unique_chunks]
    payloads = [{"source_file": file.filename, "text": t, "city": city} for t in unique_chunks]
    points = [PointStruct(id=i, vector=v.tolist(), payload=p) for i, v, p in zip(ids, unique_embeddings, payloads)]
    
    # Batched upsert
    uploaded = 0
    for start in range(0, len(points), BATCH_SIZE):
        batch = points[start : start + BATCH_SIZE]
        try:
            client.upsert(collection_name=collection_name, points=batch)
            uploaded += len(batch)
        except Exception as e:
            logging.error(f"Qdrant upsert error: {e}")
    
    logging.info(f"Uploaded {uploaded} chunks to {city} collection")
    return {"status": "ok", "chunks": uploaded, "city": city}

@app.post("/query/")
async def query(
    question: str = Form(...),
    language: str = Form("en"),
    city: str = Form("islamabad"),
    top_k: int = Form(5),
    score_threshold: float = Form(0.15)
):
    """Query city-specific collection."""
    if city not in CITY_CONFIGS:
        return JSONResponse(status_code=400, content={"error": f"Invalid city: {city}"})
    
    config = CITY_CONFIGS[city]
    collection_name = config["collection"]
    
    try:
        q_vec = embedder.encode([question], convert_to_numpy=True)[0].tolist()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Embedding failed"})
    
    try:
        results = client.search(
            collection_name=collection_name,
            query_vector=q_vec,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Search failed: {e}"})
    
    filtered = [r for r in results if getattr(r, "score", 0) >= score_threshold]
    context = "\n\n".join([r.payload.get("text", "") for r in filtered])
    
    # Query LLM
    answer = ask_mistral(context, question, city) or ask_openai(context, question, city)
    
    if not answer:
        answer = (
            f"I don't have sufficient information in the available {config['name']} documents to answer that. "
            f"Please consult the relevant {config['short_name']} manual or provide the specific document."
        )
    
    # Optional translation
    if language == "ur":
        try:
            answer = translate_to_urdu(answer)
        except Exception as e:
            logging.warning(f"Translation failed: {e}")
    
    # Citations
    unique_citations = list(set([r.payload.get("source_file", "") for r in filtered if r.payload.get("source_file")]))
    out_chunks = [{
        "score": r.score,
        "file": r.payload.get("source_file", ""),
        "chunk": r.payload.get("text", "")[:250]
    } for r in filtered]
    
    return {"answer": answer, "citations": unique_citations, "chunks": out_chunks, "city": city}

@app.post("/tts/")
async def tts(req: dict):
    """Text-to-speech endpoint."""
    text = req.get("text")
    language = req.get("language", "en")
    if TTS:
        try:
            path = TTS(text)
            return FileResponse(path, media_type="audio/mpeg")
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": "TTS failed"})
    else:
        return JSONResponse(status_code=501, content={"message": "TTS not available"})

@app.post("/stt/")
async def stt_endpoint(file: UploadFile = File(...)):
    """Speech-to-text endpoint."""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    if STT:
        try:
            transcript = STT(file_path)
            return {"transcript": transcript}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": "STT failed"})
    else:
        return JSONResponse(status_code=501, content={"message": "STT not available"})

@app.post("/sts/")
async def sts_endpoint(file: UploadFile = File(...), city: str = Form("islamabad"), top_k: int = Form(3)):
    """Speech-to-speech endpoint."""
    if not STT or not TTS:
        return JSONResponse(status_code=501, content={"message": "STT or TTS not available"})
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    try:
        question = STT(file_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "STT failed"})
    
    # Query RAG
    config = CITY_CONFIGS.get(city, CITY_CONFIGS["islamabad"])
    collection_name = config["collection"]
    
    q_vec = embedder.encode([question], convert_to_numpy=True)[0].tolist()
    results = client.search(collection_name=collection_name, query_vector=q_vec, limit=top_k, with_payload=True)
    context = "\n\n".join([r.payload.get("text", "") for r in results if r.score >= 0.4])
    answer = ask_mistral(context, question, city) or ask_openai(context, question, city) or "I don't have sufficient info."
    
    try:
        audio_path = TTS(answer)
        return FileResponse(audio_path, media_type="audio/mpeg")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "TTS failed"})
    
@app.get("/voice-token")
def voice_token(identity: str):
    now = int(time.time())

    payload = {
        "iss": LIVEKIT_API_KEY,
        "sub": identity,
        "nbf": now,
        "exp": now + 3600,
        "video": {
            "roomJoin": True,
            "room": "itp-room"
        }
    }

    token = jwt.encode(payload, LIVEKIT_API_SECRET, algorithm="HS256")

    return {
        "url": LIVEKIT_URL,
        "token": token
    }





if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag_backend_itp:app", host="0.0.0.0", port=4000, reload=True)