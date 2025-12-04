from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import secrets
import os
import requests
import json
import redis
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# ----------------------- Redis -----------------------
REDIS_URL = os.getenv("REDIS_URL")
if not REDIS_URL:
    raise RuntimeError("REDIS_URL не задан в переменных окружения Railway")

redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)


# ----------------------- FastAPI -----------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"https://(.+\.zendesk\.com|.+\.apps\.zdusercontent\.com)",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBasic()


# ----------------------- Basic Auth -----------------------
API_USERNAME = os.getenv("BASIC_AUTH_LOGIN", "admin")
API_PASSWORD = os.getenv("BASIC_AUTH_PASSWORD", "secret_password")

def check_auth(credentials: HTTPBasicCredentials = Depends(security)):
    correct_user = secrets.compare_digest(credentials.username, API_USERNAME)
    correct_pass = secrets.compare_digest(credentials.password, API_PASSWORD)

    if not (correct_user and correct_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный логин или пароль",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# ----------------------- Zendesk Settings -----------------------
ZENDESK_SUBDOMAIN = os.getenv("ZENDESK_SUBDOMAIN")
ZENDESK_EMAIL = os.getenv("ZENDESK_EMAIL")
ZENDESK_API_TOKEN = os.getenv("ZENDESK_API_TOKEN")

if not (ZENDESK_SUBDOMAIN and ZENDESK_EMAIL and ZENDESK_API_TOKEN):
    raise RuntimeError("ZENDESK_* переменные не заданы")

ZENDESK_BASE_URL = f"https://{ZENDESK_SUBDOMAIN}.zendesk.com"


# ----------------------- Gemini Settings -----------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY не задан")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.0-flash"


# ----------------------- Pydantic Models -----------------------

class TicketRequest(BaseModel):
    ticket_id: str

class SummaryStructure(BaseModel):
    issue: str
    action: str
    result: str


# ----------------------- Zendesk Logic -----------------------

def get_zendesk_audits(ticket_id: str) -> dict:
    url = f"{ZENDESK_BASE_URL}/api/v2/tickets/{ticket_id}/audits.json"
    auth = (f"{ZENDESK_EMAIL}/token", ZENDESK_API_TOKEN)

    resp = requests.get(url, auth=auth, timeout=15)
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Zendesk error: {resp.status_code}")

    return resp.json()


def extract_dialogue_from_audits(audits_json: dict) -> str:
    audits = audits_json.get("audits", [])
    messages = []

    # 1. Чат-сообщения (приоритет)
    for audit in audits:
        events = audit.get("events", [])
        for ev in events:
            if ev.get("type") == "ChatStartedEvent":
                history = ev.get("value", {}).get("history", [])
                for h in history:
                    if h.get("type") != "ChatMessage":
                        continue
                    actor = h.get("actor_type")
                    text = h.get("message", "").strip()

                    if not text:
                        continue

                    prefix = "CLIENT" if actor == "end-user" else "AGENT" if actor == "agent" else None
                    if prefix:
                        messages.append(f"{prefix}: {text}")

    # 2. Если чата нет — берем обычные комментарии
    if not messages:
        for audit in audits:
            for ev in audit.get("events", []):
                if ev.get("type") == "Comment" and ev.get("public") and ev.get("plain_body"):
                    messages.append(f"MSG: {ev['plain_body']}")

    return "\n".join(messages)


# ----------------------- AI Summarization -----------------------

def summarize_with_gemini(dialogue: str) -> dict:
    if not dialogue.strip():
        return {"issue": "Нет данных", "action": "-", "result": "-"}

    prompt = f"""
Ты — аналитик техподдержки. Проанализируй диалог:

{dialogue}

Выведи ответ СТРОГО в формате JSON с ключами:
- "issue": 1 короткое предложение
- "action": 1 короткое предложение
- "result": 1 короткое предложение

ОБЯЗАТЕЛЬНО:
- Итог всегда на ЧИСТОМ РУССКОМ языке, даже если диалог полностью на узбекском.
- Никакого markdown, только JSON.
"""

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SummaryStructure,
            )
        )

        return json.loads(response.text)

    except Exception as e:
        print("AI Error:", e)
        return {"issue": "Ошибка", "action": "Ошибка", "result": str(e)}


# ----------------------- Main Endpoint -----------------------

@app.post("/summary")
def summarize_ticket(request: TicketRequest, username: str = Depends(check_auth)):
    ticket_id = request.ticket_id

    # ---- 1. Проверяем Redis ----
    cached = redis_client.get(ticket_id)
    if cached:
        parsed = json.loads(cached)
        return {
            "ticket_id": ticket_id,
            "structured_summary": parsed["summary"],
            "dialogue": parsed["dialogue"],
            "status": "from_redis",
        }

    # ---- 2. Получаем audits ----
    audits_json = get_zendesk_audits(ticket_id)

    # ---- 3. Извлекаем диалог ----
    dialogue = extract_dialogue_from_audits(audits_json)
    if not dialogue:
        raise HTTPException(status_code=500, detail="Диалог пустой")

    # ---- 4. Генерируем summary ----
    structured = summarize_with_gemini(dialogue)

    # ---- 5. Сохраняем в Redis ----
    redis_client.set(
        ticket_id,
        json.dumps({"summary": structured, "dialogue": dialogue})
    )

    # ---- 6. Возвращаем ----
    return {
        "ticket_id": ticket_id,
        "structured_summary": structured,
        "dialogue": dialogue,
        "status": "success",
    }
