from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import secrets
import os
import requests
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],  # конкретных нет, всё через regex
    allow_origin_regex=r"https://(.+\.zendesk\.com|.+\.apps\.zdusercontent\.com)",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
security = HTTPBasic()

# -------- Basic Auth для Зендеска-вебхука --------
API_USERNAME = os.getenv("BASIC_AUTH_LOGIN", "admin")
API_PASSWORD = os.getenv("BASIC_AUTH_PASSWORD", "secret_password")

def check_auth(credentials: HTTPBasicCredentials = Depends(security)):
    is_correct_username = secrets.compare_digest(credentials.username, API_USERNAME)
    is_correct_password = secrets.compare_digest(credentials.password, API_PASSWORD)
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный логин или пароль",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# -------- Настройки Zendesk --------
ZENDESK_SUBDOMAIN = os.getenv("ZENDESK_SUBDOMAIN")  
ZENDESK_EMAIL = os.getenv("ZENDESK_EMAIL")          
ZENDESK_API_TOKEN = os.getenv("ZENDESK_API_TOKEN")

if not (ZENDESK_SUBDOMAIN and ZENDESK_EMAIL and ZENDESK_API_TOKEN):
    raise RuntimeError("Нужно задать ZENDESK_SUBDOMAIN, ZENDESK_EMAIL, ZENDESK_API_TOKEN в .env")

ZENDESK_BASE_URL = f"https://{ZENDESK_SUBDOMAIN}.zendesk.com"

# -------- Gemini --------
from google import genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Нужно задать GEMINI_API_KEY в .env")

GEMINI_MODEL = "gemini-2.5-flash"

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# -------- Модель запроса от Зендеска --------
class TicketRequest(BaseModel):
    ticket_id: str

# -------- Кеширование -------
SUMMARY_CACHE: dict[str, dict] = {}

def get_zendesk_audits(ticket_id: str) -> dict:
    """Тянем audits по тикету из Zendesk."""
    url = f"{ZENDESK_BASE_URL}/api/v2/tickets/{ticket_id}/audits.json"
    auth = (f"{ZENDESK_EMAIL}/token", ZENDESK_API_TOKEN)

    resp = requests.get(url, auth=auth, timeout=15)
    if resp.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"Zendesk вернул {resp.status_code}: {resp.text}",
        )
    return resp.json()


def extract_dialogue_from_audits(audits_json: dict) -> str:
    """
    Собираем диалог только из сообщений, где actor_type = end-user или agent.
    Берём history внутри ChatStartedEvent.
    """

    audits = audits_json.get("audits", [])
    messages = []

    for audit in audits:
        events = audit.get("events", [])
        for ev in events:
            # Исторический блок с чатом
            if ev.get("type") == "ChatStartedEvent":
                history = ev.get("value", {}).get("history", [])
                for h in history:
                    if h.get("type") != "ChatMessage":
                        continue
                    actor_type = h.get("actor_type")
                    text = h.get("message", "").strip()
                    if not text:
                        continue

                    if actor_type == "end-user":
                        prefix = "CLIENT"
                    elif actor_type == "agent":
                        prefix = "AGENT"
                    else:
                        continue

                    messages.append(f"{prefix}: {text}")

    if not messages:
        for audit in audits:
            for ev in audit.get("events", []):
                if ev.get("type") == "Comment" and ev.get("public") and ev.get("plain_body"):
                    messages.append(ev["plain_body"])

    return "\n".join(messages)


def summarize_with_gemini(dialogue: str) -> str:
    """
    Делаем саммари через Gemini.
    """
    if not dialogue.strip():
        return "Не удалось собрать диалог по тикету."

    prompt = f"""
Ты — аналитик тикетов службы поддержки.

Тебе дан диалог между клиентом и оператором:

{dialogue}

Сделай короткое саммари тикета — ровно по одному короткому предложению на каждый пункт:
1) Суть вопроса клиента — одно короткое предложение.
2) Что сделал оператор — одно короткое предложение.
3) Текущий результат/статус — одно короткое предложение.

Запрещено писать больше одного предложения на пункт.
Пиши кратко, без деталей и воды, только факты.
"""

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return (response.text or "").strip()


# ========== ОСНОВНАЯ РУЧКА ==========

@app.post("/summary")
def summarize_ticket(request: TicketRequest, username: str = Depends(check_auth)):
    ticket_id = request.ticket_id

    # 1. Кеш: если тикет уже обрабатывали — не дёргаем ИИ второй раз
    if ticket_id in SUMMARY_CACHE:
        cached = SUMMARY_CACHE[ticket_id]
        return {
            "ticket_id": ticket_id,
            "summary": cached["summary"],
            "dialogue": cached["dialogue"],
            "status": "from_cache",
        }

    # 2. Тянем audit из Zendesk
    audits_json = get_zendesk_audits(ticket_id)

    # 3. Собираем диалог
    dialogue = extract_dialogue_from_audits(audits_json)
    if not dialogue:
        raise HTTPException(
            status_code=500,
            detail="Не удалось собрать диалог (history пустой).",
        )
    # 4. Саммари через Gemini
    summary = summarize_with_gemini(dialogue)

    # 5. Сохраняем в кеш
    SUMMARY_CACHE[ticket_id] = {
        "summary": summary,
        "dialogue": dialogue,
    }

    # 6. Возвращаем ответ в том формате, который удобно Зендеску
    return {
        "ticket_id": ticket_id,
        "summary": summary,
        "dialogue": dialogue, 
        "status": "success",
    }
