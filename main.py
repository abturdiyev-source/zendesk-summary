from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import secrets
import os
import requests
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

app = FastAPI()

# Разрешаем CORS для Зендеска
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"https://(.+\.zendesk\.com|.+\.apps\.zdusercontent\.com)",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
security = HTTPBasic()

# -------- Basic Auth --------
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

# -------- Настройки --------
ZENDESK_SUBDOMAIN = os.getenv("ZENDESK_SUBDOMAIN")
ZENDESK_EMAIL = os.getenv("ZENDESK_EMAIL")
ZENDESK_API_TOKEN = os.getenv("ZENDESK_API_TOKEN")
ZENDESK_BASE_URL = f"https://{ZENDESK_SUBDOMAIN}.zendesk.com"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Нет GEMINI_API_KEY")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.0-flash" # Используем свежую быструю модель (или 1.5-flash)

# -------- Модели данных --------

class TicketRequest(BaseModel):
    ticket_id: str

# Модель для структуры саммари (чтобы мы всегда отдавали одинаковый формат)
class SummaryStructure(BaseModel):
    issue: str
    action: str
    result: str

# -------- Кеш --------
# Храним теперь не строку, а словарь с полями
SUMMARY_CACHE: dict[str, dict] = {}

# -------- Логика Zendesk --------

def get_zendesk_audits(ticket_id: str) -> dict:
    url = f"{ZENDESK_BASE_URL}/api/v2/tickets/{ticket_id}/audits.json"
    auth = (f"{ZENDESK_EMAIL}/token", ZENDESK_API_TOKEN)
    resp = requests.get(url, auth=auth, timeout=15)
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Zendesk error: {resp.status_code}")
    return resp.json()

def extract_dialogue_from_audits(audits_json: dict) -> str:
    # (Твоя логика без изменений, она хорошая)
    audits = audits_json.get("audits", [])
    messages = []
    
    # Сначала ищем чаты
    for audit in audits:
        events = audit.get("events", [])
        for ev in events:
            if ev.get("type") == "ChatStartedEvent":
                history = ev.get("value", {}).get("history", [])
                for h in history:
                    if h.get("type") != "ChatMessage": continue
                    actor = h.get("actor_type")
                    text = h.get("message", "").strip()
                    if not text: continue
                    
                    prefix = "CLIENT" if actor == "end-user" else "AGENT" if actor == "agent" else None
                    if prefix:
                        messages.append(f"{prefix}: {text}")
    
    # Если чатов нет, берем обычные комменты
    if not messages:
        for audit in audits:
            for ev in audit.get("events", []):
                if ev.get("type") == "Comment" and ev.get("public") and ev.get("plain_body"):
                    messages.append(f"MSG: {ev['plain_body']}")

    return "\n".join(messages)

# -------- Логика AI (Обновленная) --------

def summarize_with_gemini(dialogue: str) -> dict:
    """
    Возвращает словарь {"issue": "...", "action": "...", "result": "..."}
    """
    if not dialogue.strip():
        return {"issue": "Нет данных", "action": "-", "result": "-"}

    # Промпт теперь просит JSON
    prompt = f"""
    Ты — аналитик техподдержки. Проанализируй диалог:
    
    {dialogue}
    
    Выведи ответ СТРОГО в формате JSON с тремя ключами:
    "issue": (Суть проблемы клиента, 1 короткое предложение)
    "action": (Что конкретно сделал агент, 1 короткое предложение)
    "result": (Итог: решено/в процессе/ждет ответа, 1 короткое предложение)
    
    Не добавляй никакого markdown форматирования (```json), только чистый JSON.
    """

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", # <--- ГЛАВНАЯ МАГИЯ
                response_schema=SummaryStructure # Опционально, для строгой типизации
            )
        )
        
        # Gemini вернет JSON строку, парсим её в Python dict
        result_json = json.loads(response.text)
        return result_json
        
    except Exception as e:
        print(f"AI Error: {e}")
        return {
            "issue": "Ошибка генерации",
            "action": "Проверьте логи", 
            "result": str(e)
        }

# ========== РУЧКА ==========

@app.post("/summary")
def summarize_ticket(request: TicketRequest, username: str = Depends(check_auth)):
    ticket_id = request.ticket_id

    # 1. Кеш
    if ticket_id in SUMMARY_CACHE:
        cached = SUMMARY_CACHE[ticket_id]
        return {
            "ticket_id": ticket_id,
            "structured_summary": cached["summary"],
            "dialogue": cached["dialogue"], 
            "status": "from_cache",
        }

    # 2. Получаем audits
    audits_json = get_zendesk_audits(ticket_id)

    # 3. Диалог
    dialogue = extract_dialogue_from_audits(audits_json)
    if not dialogue:
        raise HTTPException(
            status_code=500,
            detail="Не удалось собрать диалог по тикету.",
        )

    # 4. Генерация саммари
    structured = summarize_with_gemini(dialogue)

    # 5. Кладём в кеш
    SUMMARY_CACHE[ticket_id] = {
        "summary": structured,
        "dialogue": dialogue,
    }

    # 6. Возвращаем И САММАРИ, И ДИАЛОГ
    return {
        "ticket_id": ticket_id,
        "structured_summary": structured,
        "dialogue": dialogue,  
        "status": "success",
    }
