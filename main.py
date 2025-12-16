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

# 1. Загружаем настройки
load_dotenv()

app = FastAPI(title="Zendesk Auto-QA Service")

# 2. Настраиваем CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_origin_regex=r"https://(.+\.zendesk\.com|.+\.apps\.zdusercontent\.com)",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBasic()

# --- ПРОВЕРКА ENV ---
REQUIRED_VARS = ["ZENDESK_SUBDOMAIN", "ZENDESK_EMAIL", "ZENDESK_API_TOKEN", "GEMINI_API_KEY"]
missing = [v for v in REQUIRED_VARS if not os.getenv(v)]
if missing:
    print(f"⚠️  FATAL: В .env не хватает ключей: {', '.join(missing)}")

# --- КОНФИГУРАЦИЯ ---
API_USER = os.getenv("BASIC_AUTH_LOGIN", "admin")
API_PASS = os.getenv("BASIC_AUTH_PASSWORD", "secret")

ZD_URL = f"https://{os.getenv('ZENDESK_SUBDOMAIN')}.zendesk.com"
ZD_AUTH = (f"{os.getenv('ZENDESK_EMAIL')}/token", os.getenv('ZENDESK_API_TOKEN'))

# ИИ
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL_SUMMARY = "gemini-2.0-flash" # Быстрая для саммари
GEMINI_MODEL_QA = "gemini-2.5-flash"      # Умная для оценки

# Redis
REDIS_URL = os.getenv("REDIS_URL")
try:
    if REDIS_URL:
        r = redis.from_url(REDIS_URL, decode_responses=True)
    else:
        r = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=0, decode_responses=True
        )
    r.ping()
    print("✅ Redis подключен")
except Exception as e:
    print(f"⚠️ Redis недоступен: {e}. Работаем без кеша.")
    r = None

# --- МОДЕЛИ ДАННЫХ ---
class TicketRequest(BaseModel):
    ticket_id: str
    class Config:
        json_schema_extra = {"example": {"ticket_id": "21579460"}}

# Модель 1: Саммари
class TicketSummary(BaseModel):
    ticket_id: str
    assignee_id: int | str | None = None
    agent_name: str | None = "Unknown"
    
    issue: str
    action: str
    result: str
    
    status: str | None = None

    class Config:
        json_schema_extra = {
            "example": {
                "ticket_id": "21579460",
                "assignee_id": 12345,
                "agent_name": "Иван Иванов",
                "issue": "Клиент не мог войти",
                "action": "Сбросил пароль",
                "result": "Успех",
                "status": "generated_new"
            }
        }

# Модель 2: Оценка (QA)
class TicketEvaluation(BaseModel):
    ticket_id: str
    assignee_id: int | str | None = None
    agent_name: str | None = "Unknown"
    
    language: str
    tov_score: int
    solution_score: int
    errors: list[str]
    next_action: str
    
    analyzed_at: str | None = None
    status: str | None = None

    class Config:
        json_schema_extra = {
            "example": {
                "ticket_id": "21579460",
                "assignee_id": 12345,
                "agent_name": "Иван Иванов",
                "language": "ru",
                "tov_score": 5,
                "solution_score": 5,
                "errors": [],
                "next_action": "Молодец",
                "analyzed_at": "2025-12-16T15:30:00",
                "status": "generated_new"
            }
        }

# --- ЛОГИКА ---

# --- ЗАГРУЗКА TOV (Новая логика) ---
def load_tov_rules():
    """Читает файл с правилами, если он есть"""
    try:
        with open("tov_rules.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print("⚠️ Внимание: Файл tov_rules.md не найден! Используем общие правила.")
        return "Правила не заданы. Оценивай на основе здравого смысла и вежливости."

TOV_RULES = load_tov_rules() # Загружаем 1 раз при старте сервера

def check_auth(creds: HTTPBasicCredentials = Depends(security)):
    if not (secrets.compare_digest(creds.username, API_USER) and 
            secrets.compare_digest(creds.password, API_PASS)):
        raise HTTPException(status_code=401, detail="Auth Error")
    return creds.username

def get_zendesk_data(ticket_id: str):
    """
    ВАЖНОЕ ИСПРАВЛЕНИЕ: Делаем 2 отдельных запроса.
    1. Тикет + Юзеры (для имени агента)
    2. Аудиты (для диалога)
    """
    print(f"📡 ZENDESK: Качаем метаданные тикета {ticket_id}...")
    
    # Запрос 1: Метаданные
    url_ticket = f"{ZD_URL}/api/v2/tickets/{ticket_id}.json?include=users"
    try:
        resp_ticket = requests.get(url_ticket, auth=ZD_AUTH, timeout=10)
        if resp_ticket.status_code == 404:
            raise HTTPException(status_code=404, detail="Ticket not found")
        if resp_ticket.status_code != 200:
            print(f"❌ Ошибка Ticket API: {resp_ticket.text}")
            raise HTTPException(status_code=500, detail="Zendesk API Error")
        ticket_data = resp_ticket.json()
    except Exception as e:
        print(f"❌ Сетевая ошибка (Ticket): {e}")
        raise HTTPException(status_code=500, detail="Network Error")

    # Запрос 2: История (Аудиты) - отдельно, чтобы не обрезалось!
    print(f"📡 ZENDESK: Качаем полную историю (audits)...")
    url_audits = f"{ZD_URL}/api/v2/tickets/{ticket_id}/audits.json"
    try:
        resp_audits = requests.get(url_audits, auth=ZD_AUTH, timeout=15)
        if resp_audits.status_code != 200:
            print(f"⚠️ Ошибка Audits API: {resp_audits.text}. Диалог будет пуст.")
            audits_list = []
        else:
            audits_list = resp_audits.json().get("audits", [])
    except Exception as e:
        print(f"⚠️ Сетевая ошибка (Audits): {e}")
        audits_list = []

    # Склеиваем результат
    return {
        "ticket": ticket_data.get("ticket", {}),
        "users": ticket_data.get("users", []),
        "audits": audits_list
    }

def parse_ticket_data(data: dict) -> tuple[str, str, int | str | None]:
    """Разбирает JSON: находит диалог и агента"""
    print("🔍 PARSER: Начинаем разбор...")
    ticket = data.get("ticket", {})
    users = data.get("users", [])
    audits = data.get("audits", [])
    
    # 1. Ищем ID Агента
    assignee = ticket.get("assignee") or ticket.get("assignee_id")
    
    # Если в шапке нет, ищем в истории (последнее назначение)
    if not assignee:
        for audit in reversed(audits):
            if audit.get("assignee"): assignee = audit.get("assignee"); break
            if audit.get("assignee_id"): assignee = audit.get("assignee_id"); break
            for ev in audit.get("events", []):
                if ev.get("field_name") in ["assignee", "assignee_id"] and ev.get("value"):
                    assignee = ev.get("value"); break
            if assignee: break
            
    # 2. Ищем Имя Агента
    agent_name = "Unknown Agent"
    if assignee:
        try:
            target_id = int(assignee)
            for u in users:
                if u["id"] == target_id:
                    agent_name = u["name"]
                    break
        except: pass 
        
    print(f"🔍 PARSER: Агент: {agent_name} (ID: {assignee})")
    print(f"🔍 PARSER: Всего блоков аудита доступно: {len(audits)}")

    # 3. Собираем Диалог (Улучшенная логика)
    messages = []
    user_map = {u["id"]: u["name"] for u in users}
    IGNORE = ["Mutaxassisni chaqirish", "Main Menu", "Start Chat", "Bot started"]

    for audit in audits:
        for ev in audit.get("events", []):
            event_type = ev.get("type")
            
            # Тип А: Чаты (Messaging)
            if event_type == "ChatStartedEvent":
                history = ev.get("value", {}).get("history", [])
                if not history and "history" in ev: 
                    history = ev["history"]
                
                for h in history:
                    if h.get("type") != "ChatMessage": continue
                    msg = h.get("message", "")
                    if msg is None: msg = ""
                    msg = str(msg).strip()
                    
                    if not msg or any(x in msg for x in IGNORE): continue
                    
                    role = h.get("actor_type") # end-user / agent
                    d_name = h.get("name") or h.get("actor_name") or "User"
                    
                    if h.get("author_id") and h.get("author_id") in user_map:
                        d_name = user_map[h.get("author_id")]

                    prefix = f"CLIENT ({d_name})" if role == "end-user" else f"AGENT ({d_name})"
                    messages.append(f"{prefix}: {msg}")
            
            # Тип Б: Почта/Комменты
            elif event_type == "Comment":
                is_public = ev.get("public", False)
                if is_public:
                    body = ev.get("plain_body") or ev.get("body")
                    if body:
                        author_id = ev.get("author_id")
                        author_name = user_map.get(author_id, "AGENT")
                        messages.append(f"{author_name}: {body}")

    dialogue = "\n".join(messages)
    print(f"📝 PARSER: Итого сообщений в диалоге: {len(messages)}")
    return dialogue, agent_name, assignee

# --- ФУНКЦИИ ИИ (РАЗДЕЛЕННЫЕ) ---

# --- ОБНОВЛЕННАЯ ФУНКЦИЯ ОЦЕНКИ ---
def run_evaluation_ai(ticket_id: str, dialogue: str) -> dict:
    print("🤖 AI (QA): Отправка запроса с ToV...")
    
    # Вставляем правила (TOV_RULES) прямо в промпт
    prompt = f"""
    Ты — строгий QA аналитик поддержки. Твоя цель — проверить соответствие диалога регламенту.
    
    === РЕГЛАМЕНТ КОМПАНИИ (ToV) ===
    {TOV_RULES}
    ================================
    
    ВАЖНО:
    1. Оценивай СТРОГО по тексту регламента выше.
    2. Если агент нарушил конкретный пункт из регламента, укажи это в errors.
    3. Отвечай на РУССКОМ языке.
    
    === ДИАЛОГ ДЛЯ ПРОВЕРКИ ===
    {dialogue}
    ===========================
    
    Выведи JSON:
    - language (ru/uz/en)
    - tov_score (0-5, где 5 - полное соблюдение регламента)
    - solution_score (0-5)
    - errors (список нарушений со ссылкой на пункты регламента)
    - next_action (совет агенту)
    """
    
    try:
        resp = gemini_client.models.generate_content(
            model=GEMINI_MODEL_QA,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=TicketEvaluation)
        )
        res = json.loads(resp.text)
        res["analyzed_at"] = str(datetime.now())
        return res
    except Exception as e:
        print(f"❌ AI ERROR: {e}")
        return {
            "ticket_id": ticket_id, "language": "err", "tov_score": 0, "solution_score": 0,
            "errors": [str(e)], "next_action": "-", "analyzed_at": str(datetime.now())
        }


def run_summary_ai(ticket_id: str, dialogue: str) -> dict:
    print("🤖 AI (Summary): Отправка...")
    prompt = f"""
    Ты — помощник оператора. Сделай краткое саммари тикета.
    ВАЖНО: ОТВЕЧАЙ СТРОГО НА РУССКОМ ЯЗЫКЕ.
    Диалог:
    {dialogue}
    JSON (на русском):
    - issue: Суть проблемы (1 предл)
    - action: Что сделал оператор (1 предл)
    - result: Итог (1 предл)
    """
    try:
        resp = gemini_client.models.generate_content(
            model=GEMINI_MODEL_SUMMARY,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=TicketSummary)
        )
        return json.loads(resp.text)
    except Exception as e:
        print(f"❌ AI ERROR: {e}")
        return {"ticket_id": ticket_id, "issue": "Error", "action": "-", "result": str(e)}

def run_evaluation_ai(ticket_id: str, dialogue: str) -> dict:
    print("🤖 AI (QA): Отправка...")
    prompt = f"""
    Ты — QA аналитик. Оцени качество диалога.
    ВАЖНО: ОТВЕЧАЙ СТРОГО НА РУССКОМ ЯЗЫКЕ.
    Диалог: {dialogue}
    JSON (на русском):
    - language (ru/uz/en)
    - tov_score (0-5)
    - solution_score (0-5)
    - errors (список)
    - next_action (совет)
    """
    try:
        resp = gemini_client.models.generate_content(
            model=GEMINI_MODEL_QA,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=TicketEvaluation)
        )
        res = json.loads(resp.text)
        res["analyzed_at"] = str(datetime.now())
        return res
    except Exception as e:
        print(f"❌ AI ERROR: {e}")
        return {
            "ticket_id": ticket_id, "language": "err", "tov_score": 0, "solution_score": 0,
            "errors": [str(e)], "next_action": "-", "analyzed_at": str(datetime.now())
        }

# --- РУЧКИ ---

@app.post("/summary", response_model=TicketSummary)
def get_summary(req: TicketRequest, user: str = Depends(check_auth)):
    tid = req.ticket_id
    if r:
        cached = r.get(f"summary:{tid}")
        if cached: return {**json.loads(cached), "status": "from_cache"}

    data = get_zendesk_data(tid)
    dialogue, agent, aid = parse_ticket_data(data)
    
    if not dialogue:
        res = {"ticket_id": tid, "assignee_id": aid, "agent_name": agent, "issue": "Нет диалога", "action": "-", "result": "-", "status": "empty"}
        # Не кешируем ошибку надолго, если проблема была сетевой
        return res

    result = run_summary_ai(tid, dialogue)
    result.update({"ticket_id": tid, "assignee_id": aid, "agent_name": agent, "status": "generated_new"})
    if r: r.set(f"summary:{tid}", json.dumps(result))
    return result

@app.post("/evaluate", response_model=TicketEvaluation)
def evaluate_ticket(req: TicketRequest, user: str = Depends(check_auth)):
    tid = req.ticket_id
    if r:
        cached = r.get(f"qa:{tid}")
        if cached: return {**json.loads(cached), "status": "from_cache"}

    data = get_zendesk_data(tid)
    dialogue, agent, aid = parse_ticket_data(data)

    if not dialogue:
        res = {"ticket_id": tid, "assignee_id": aid, "agent_name": agent, "language": "n/a", "tov_score": 0, "solution_score": 0, "errors": ["Empty"], "next_action": "-", "status": "empty"}
        return res

    result = run_evaluation_ai(tid, dialogue)
    result.update({"ticket_id": tid, "assignee_id": aid, "agent_name": agent, "status": "generated_new"})
    if r: r.set(f"qa:{tid}", json.dumps(result))
    return result

@app.get("/analytics/errors")
def get_errors(user: str = Depends(check_auth)):
    if not r: return {"error": "No Redis"}
    rows = []
    for k in r.scan_iter("qa:*"):
        val = r.get(k)
        if val:
            d = json.loads(val)
            if d.get("tov_score", 5) < 4 or d.get("solution_score", 5) < 4 or d.get("errors"):
                rows.append(d)
    return {"count": len(rows), "data": rows}