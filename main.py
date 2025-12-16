from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import secrets
import os
import requests
import json
import redis  # <--- Новая библиотека
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

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

# -------- Настройки API --------
ZENDESK_SUBDOMAIN = os.getenv("ZENDESK_SUBDOMAIN")
ZENDESK_EMAIL = os.getenv("ZENDESK_EMAIL")
ZENDESK_API_TOKEN = os.getenv("ZENDESK_API_TOKEN")
ZENDESK_BASE_URL = f"https://{ZENDESK_SUBDOMAIN}.zendesk.com"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.0-flash" 

# -------- REDIS ПОДКЛЮЧЕНИЕ --------
# Обычно хост 'localhost', порт 6379, db 0
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# Создаем клиент
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)

# -------- Функции для Redis --------

def get_summary_from_redis(ticket_id: str):
    """Ищет ключ 'ticket:12345'."""
    key = f"ticket:{ticket_id}"
    data_json = r.get(key)
    
    if data_json:
        # Redis возвращает строку, превращаем её обратно в словарь
        return json.loads(data_json)
    return None

def save_summary_to_redis(ticket_id: str, data: dict):
    """
    Сохраняет саммари. 
    Мы добавляем поле created_at сами перед сохранением.
    """
    key = f"ticket:{ticket_id}"
    
    # Добавим таймстемп, чтобы знать когда создали
    data["created_at"] = str(os.getenv("Time_Now", "Just now")) 
    
    # Превращаем словарь в строку JSON
    data_str = json.dumps(data, ensure_ascii=False)
    
    r.set(key, data_str)
    # Если нужно, чтобы кеш протухал через неделю (604800 сек), раскомментируй:
    # r.expire(key, 604800) 

# -------- Модели --------
class TicketRequest(BaseModel):
    ticket_id: str

class SummaryStructure(BaseModel):
    issue: str
    action: str
    result: str

# -------- Логика Zendesk (стандартная) --------
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
                    if prefix: messages.append(f"{prefix}: {text}")
    if not messages:
        for audit in audits:
            for ev in audit.get("events", []):
                if ev.get("type") == "Comment" and ev.get("public") and ev.get("plain_body"):
                    messages.append(f"MSG: {ev['plain_body']}")
    return "\n".join(messages)

# -------- Логика AI --------
def summarize_with_gemini(dialogue: str) -> dict:
    if not dialogue.strip():
        return {"issue": "Нет данных", "action": "-", "result": "-"}
    prompt = f"""
    Ты — аналитик техподдержки. Проанализируй диалог: {dialogue}
    Выведи JSON: "issue", "action", "result" (по 1 предложению).
    """
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SummaryStructure
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"AI Error: {e}")
        return {"issue": "Ошибка", "action": "Ошибка", "result": str(e)}

# ========== РУЧКИ ==========

@app.post("/summary")
def summarize_ticket(request: TicketRequest, username: str = Depends(check_auth)):
    ticket_id = request.ticket_id

    # 1. Сначала Redis
    cached = get_summary_from_redis(ticket_id)
    if cached:
        print(f"REDIS Hit for {ticket_id}")
        return {
            "ticket_id": ticket_id,
            "structured_summary": cached,
            "status": "from_redis"
        }

    # 2. Если нет - работаем
    audits = get_zendesk_audits(ticket_id)
    dialogue = extract_dialogue_from_audits(audits)
    
    if not dialogue:
        return {"ticket_id": ticket_id, "error": "Empty dialogue", "status": "failed"}

    structured_data = summarize_with_gemini(dialogue)

    # 3. Сохраняем в Redis
    save_summary_to_redis(ticket_id, structured_data)

    return {
        "ticket_id": ticket_id,
        "structured_summary": structured_data,
        "status": "generated_new"
    }

# НОВАЯ РУЧКА ДЛЯ ВЫГРУЗКИ (SCAN)
@app.get("/history")
def get_all_redis_history(username: str = Depends(check_auth)):
    """
    Сканирует весь Redis в поиске ключей 'ticket:*'
    и собирает их значения.
    """
    all_data = []
    # scan_iter - безопасный способ перебрать ключи, не вешая сервер
    for key in r.scan_iter("ticket:*"):
        val_json = r.get(key)
        if val_json:
            data = json.loads(val_json)
            # Добавим сам ID тикета внутрь для удобства чтения
            data["ticket_id_key"] = key 
            all_data.append(data)
            
    return {"count": len(all_data), "data": all_data}