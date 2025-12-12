from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import secrets
import os
import requests
import json
import redis
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

app = FastAPI(title="Zendesk Service: Summary & QA")

# Разрешаем CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"https://(.+\.zendesk\.com|.+\.apps\.zdusercontent\.com)",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBasic()

# ================= НАСТРОЙКИ =================

API_USERNAME = os.getenv("BASIC_AUTH_LOGIN", "admin")
API_PASSWORD = os.getenv("BASIC_AUTH_PASSWORD", "secret")

ZENDESK_SUBDOMAIN = os.getenv("ZENDESK_SUBDOMAIN")
ZENDESK_EMAIL = os.getenv("ZENDESK_EMAIL")
ZENDESK_API_TOKEN = os.getenv("ZENDESK_API_TOKEN")
ZENDESK_BASE_URL = f"https://{ZENDESK_SUBDOMAIN}.zendesk.com"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Не найден GEMINI_API_KEY")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# --- КОНФИГУРАЦИЯ МОДЕЛЕЙ ---
# Для быстрого саммари используем легкую 2.0
GEMINI_MODEL_SUMMARY = "gemini-2.0-flash" 
# Для сложного анализа используем умную 2.5
GEMINI_MODEL_QA = "gemini-2.5-flash"

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
try:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
    r.ping()
except redis.ConnectionError:
    print("WARNING: Нет соединения с Redis!")
    r = None

# ================= МОДЕЛИ ДАННЫХ =================

class TicketRequest(BaseModel):
    ticket_id: str

# МОДЕЛЬ 1: Чистое Саммари (для Агента)
class TicketSummary(BaseModel):
    ticket_id: str
    issue: str      # Суть
    action: str     # Действие
    result: str     # Итог
    status: str | None = None # from_cache / generated_new

# МОДЕЛЬ 2: Оценка качества (для Аналитика)
class TicketEvaluation(BaseModel):
    ticket_id: str
    language: str
    tov_score: int       # 0-5
    solution_score: int  # 0-5
    errors: list[str]    # Список ошибок
    next_action: str     # Рекомендация
    analyzed_at: str | None = None
    status: str | None = None

# ================= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (ZENDESK) =================

def check_auth(credentials: HTTPBasicCredentials = Depends(security)):
    is_correct_username = secrets.compare_digest(credentials.username, API_USERNAME)
    is_correct_password = secrets.compare_digest(credentials.password, API_PASSWORD)
    if not (is_correct_username and is_correct_password):
        raise HTTPException(status_code=401, detail="Auth Error")
    return credentials.username

def get_zendesk_audits(ticket_id: str) -> dict:
    url = f"{ZENDESK_BASE_URL}/api/v2/tickets/{ticket_id}/audits.json"
    auth = (f"{ZENDESK_EMAIL}/token", ZENDESK_API_TOKEN)
    resp = requests.get(url, auth=auth, timeout=15)
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Zendesk error: {resp.status_code}")
    return resp.json()

def extract_dialogue(audits_json: dict) -> str:
    audits = audits_json.get("audits", [])
    messages = []
    IGNORE_PHRASES = ["Mutaxassisni chaqirish", "Позвать специалиста", "Main Menu", "Start Chat", "Bot started"]

    for audit in audits:
        events = audit.get("events", [])
        for ev in events:
            if ev.get("type") == "ChatStartedEvent":
                history = ev.get("value", {}).get("history", [])
                for h in history:
                    if h.get("type") != "ChatMessage": continue
                    text = h.get("message", "").strip()
                    if not text or any(p in text for p in IGNORE_PHRASES): continue
                    
                    actor = h.get("actor_type")
                    prefix = "CLIENT" if actor == "end-user" else "AGENT" if actor == "agent" else None
                    if prefix: messages.append(f"{prefix}: {text}")
                        
    if not messages:
         for audit in audits:
            for ev in audit.get("events", []):
                if ev.get("type") == "Comment" and ev.get("public"):
                    body = ev.get("plain_body", "")
                    if body: messages.append(f"MSG: {body}")

    return "\n".join(messages)

# ================= ФУНКЦИИ ИИ (ДВЕ РАЗНЫЕ) =================

# 1. Генерация Саммари (Используем модель 2.0)
def generate_summary_ai(ticket_id: str, dialogue: str) -> dict:
    prompt = f"""
    Ты — помощник оператора поддержки. Твоя задача — очень кратко резюмировать тикет.
    Диалог:
    {dialogue}
    
    Заполни JSON (строго по 1 предложению на поле):
    - issue: В чем была проблема клиента?
    - action: Что конкретно сделал оператор?
    - result: Чем всё закончилось (решено/нет)?
    """
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL_SUMMARY, # <-- Model 2.0 Flash
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TicketSummary
            )
        )
        return json.loads(response.text)
    except Exception as e:
        return {"ticket_id": ticket_id, "issue": "Error", "action": "-", "result": str(e)}

# 2. Генерация Оценки (Используем модель 2.5)
def generate_evaluation_ai(ticket_id: str, dialogue: str) -> dict:
    prompt = f"""
    Ты — строгий QA аналитик. Проверь качество работы агента.
    Диалог: {dialogue}
    
    Правила оценки:
    1. Language: ru/uz/mixed.
    2. TOV Score (0-5): Вежливость, эмпатия.
    3. Solution Score (0-5): Правильность решения, наличие шагов.
    4. Errors: Список ошибок (грубость, игнор вопросов). Если нет - пустой список.
    5. Next Action: Совет агенту.
    """
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL_QA, # <-- Model 2.5 Flash
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TicketEvaluation
            )
        )
        res = json.loads(response.text)
        res["analyzed_at"] = datetime.now().isoformat()
        return res
    except Exception as e:
        return {
            "ticket_id": ticket_id, "language": "err", "tov_score": 0, "solution_score": 0,
            "errors": [str(e)], "next_action": "-", "analyzed_at": datetime.now().isoformat()
        }

# ================= РУЧКИ (ENDPOINTS) =================

# --- РУЧКА 1: ПРОСТОЕ САММАРИ (Для приложения) ---
@app.post("/summary", response_model=TicketSummary)
def get_summary(request: TicketRequest, username: str = Depends(check_auth)):
    ticket_id = request.ticket_id
    
    # Redis Key: summary:123
    if r:
        cached = r.get(f"summary:{ticket_id}")
        if cached:
            data = json.loads(cached)
            data["status"] = "from_cache"
            return data

    audits = get_zendesk_audits(ticket_id)
    dialogue = extract_dialogue(audits)
    
    if not dialogue:
        res = {"ticket_id": ticket_id, "issue": "Нет диалога", "action": "-", "result": "-", "status": "empty"}
        if r: r.set(f"summary:{ticket_id}", json.dumps(res))
        return res

    result = generate_summary_ai(ticket_id, dialogue)
    result["status"] = "generated_new"
    
    if r: r.set(f"summary:{ticket_id}", json.dumps(result))
    return result


# --- РУЧКА 2: ОЦЕНКА КАЧЕСТВА (Для аналитики) ---
@app.post("/evaluate", response_model=TicketEvaluation)
def evaluate_ticket(request: TicketRequest, username: str = Depends(check_auth)):
    ticket_id = request.ticket_id

    # Redis Key: qa:123 (Отдельный кеш!)
    if r:
        cached = r.get(f"qa:{ticket_id}")
        if cached:
            data = json.loads(cached)
            data["status"] = "from_cache"
            return data

    audits = get_zendesk_audits(ticket_id)
    dialogue = extract_dialogue(audits)

    if not dialogue:
        # Для аналитики пустой диалог - это тоже результат
        res = {
            "ticket_id": ticket_id, "language": "n/a", "tov_score": 0, "solution_score": 0,
            "errors": ["Empty Dialogue"], "next_action": "-", "analyzed_at": datetime.now().isoformat(), "status": "empty"
        }
        if r: r.set(f"qa:{ticket_id}", json.dumps(res))
        return res

    result = generate_evaluation_ai(ticket_id, dialogue)
    result["status"] = "generated_new"

    if r: r.set(f"qa:{ticket_id}", json.dumps(result))
    return result


# --- РУЧКА 3: ТАБЛИЦА ОШИБОК (Берет данные из evaluations) ---
@app.get("/analytics/errors")
def get_errors_table(username: str = Depends(check_auth)):
    if not r: return {"error": "No Redis"}
    
    rows = []
    # Сканируем только ключи qa:*
    for key in r.scan_iter("qa:*"):
        val = r.get(key)
        if val:
            data = json.loads(val)
            # Фильтр: есть ошибки или низкая оценка
            has_errors = len(data.get("errors", [])) > 0
            low_score = (data.get("tov_score", 5) < 4) or (data.get("solution_score", 5) < 4)
            
            if has_errors or low_score:
                rows.append(data)
    
    return {"count": len(rows), "data": rows}