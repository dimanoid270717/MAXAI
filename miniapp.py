import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path
from urllib.parse import parse_qsl

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

router = APIRouter()
INDEX = Path(__file__).parent / "miniapp" / "index.html"


class AuthBody(BaseModel):
    init_data: str = Field(min_length=1, max_length=10000)


class GenerateBody(BaseModel):
    kind: str
    prompt: str = Field(min_length=5, max_length=20000)
    resolution: str


class PaymentBody(BaseModel):
    credits: int


def _db():
    from main import db_connect, initialize_database
    initialize_database()
    with db_connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS miniapp_sessions (
                token_hash TEXT PRIMARY KEY,
                max_user_id BIGINT NOT NULL REFERENCES users(max_user_id) ON DELETE CASCADE,
                expires_at TIMESTAMPTZ NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS miniapp_sessions_expiry_idx
            ON miniapp_sessions(expires_at)
        """)
    return db_connect


def _validate_init_data(init_data: str, bot_token: str):
    try:
        pairs = parse_qsl(init_data, keep_blank_values=True, strict_parsing=True)
        keys = [key for key, _ in pairs]
        if len(keys) != len(set(keys)) or keys.count("hash") != 1:
            return None
        data = dict(pairs)
        received_hash = data.pop("hash")
        auth_date = int(data["auth_date"])
        now = int(time.time())
        if auth_date > now + 60 or now - auth_date > 3600:
            return None
        check_string = "\n".join(f"{k}={v}" for k, v in sorted(data.items()))
        secret_key = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
        calculated = hmac.new(secret_key, check_string.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(calculated, received_hash):
            return None
        user = json.loads(data["user"])
        if not isinstance(user, dict) or not user.get("id"):
            return None
        return user
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _current_user(authorization: str | None):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Требуется авторизация MAX")
    token = authorization[7:].strip()
    if len(token) < 32:
        raise HTTPException(status_code=401, detail="Недействительная сессия")
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    db_connect = _db()
    with db_connect() as conn:
        row = conn.execute("""
            SELECT max_user_id FROM miniapp_sessions
            WHERE token_hash=%s AND expires_at > NOW()
        """, (token_hash,)).fetchone()
    if not row:
        raise HTTPException(status_code=401, detail="Сессия истекла. Открой приложение заново.")
    return int(row["max_user_id"])


@router.get("/miniapp")
async def miniapp_home():
    if not INDEX.is_file():
        raise HTTPException(status_code=503, detail="Интерфейс Mini App ещё не установлен")
    return FileResponse(INDEX)


@router.post("/miniapp/api/auth")
async def miniapp_auth(body: AuthBody):
    bot_token = os.getenv("MAX_BOT_TOKEN", "").strip()
    if not bot_token:
        raise HTTPException(status_code=503, detail="Авторизация MAX пока не настроена")
    user = _validate_init_data(body.init_data, bot_token)
    if not user:
        raise HTTPException(status_code=401, detail="Не удалось подтвердить подпись MAX. Открой приложение из чата с ботом.")
    user_id = int(user["id"])
    from main import save_user
    save_user(user_id, user.get("username"), user.get("first_name"))
    token = secrets.token_urlsafe(36)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    db_connect = _db()
    with db_connect() as conn:
        conn.execute("DELETE FROM miniapp_sessions WHERE expires_at < NOW()")
        conn.execute("""
            INSERT INTO miniapp_sessions(token_hash, max_user_id, expires_at)
            VALUES (%s, %s, NOW() + INTERVAL '12 hours')
        """, (token_hash, user_id))
        row = conn.execute("SELECT balance FROM users WHERE max_user_id=%s", (user_id,)).fetchone()
    return {
        "token": token,
        "user": {"id": user_id, "first_name": user.get("first_name") or "Пользователь"},
        "balance": int(row["balance"]) if row else 0,
        "generation_enabled": os.getenv("ENABLE_PAID_GENERATION", "false").lower() in ("1", "true", "yes", "on"),
        "payments_configured": bool(os.getenv("YOOKASSA_SHOP_ID") and os.getenv("YOOKASSA_SECRET_KEY")),
        "payments_enabled": os.getenv("ENABLE_PAYMENTS", "false").strip().lower() in ("1", "true", "yes", "on"),
    }


@router.get("/miniapp/api/me")
async def miniapp_me(authorization: str | None = Header(default=None)):
    user_id = _current_user(authorization)
    db_connect = _db()
    with db_connect() as conn:
        row = conn.execute("SELECT first_name, balance FROM users WHERE max_user_id=%s", (user_id,)).fetchone()
    return {
        "user": {"id": user_id, "first_name": row["first_name"] if row else "Пользователь"},
        "balance": int(row["balance"]) if row else 0,
        "generation_enabled": os.getenv("ENABLE_PAID_GENERATION", "false").lower() in ("1", "true", "yes", "on"),
        "payments_configured": bool(os.getenv("YOOKASSA_SHOP_ID") and os.getenv("YOOKASSA_SECRET_KEY")),
    }


@router.get("/miniapp/api/history")
async def miniapp_history(authorization: str | None = Header(default=None)):
    user_id = _current_user(authorization)
    db_connect = _db()
    with db_connect() as conn:
        rows = conn.execute("""
            SELECT id, generation_type, prompt, status, resolution,
                   price_credits, result_url, created_at
            FROM generation_history WHERE max_user_id=%s
            ORDER BY created_at DESC, id DESC LIMIT 20
        """, (user_id,)).fetchall()
    return {"items": [{
        "id": int(r["id"]), "kind": r["generation_type"], "prompt": r["prompt"] or "",
        "status": r["status"], "resolution": r["resolution"],
        "price": r["price_credits"], "result_url": r["result_url"],
        "created_at": r["created_at"].isoformat()
    } for r in rows]}


@router.post("/miniapp/api/generate")
async def miniapp_generate(body: GenerateBody, authorization: str | None = Header(default=None)):
    user_id = _current_user(authorization)
    from main import GENERATION_PRICES, db_connect, launch_generation
    if os.getenv("ENABLE_PAID_GENERATION", "false").strip().lower() not in ("1", "true", "yes", "on"):
        raise HTTPException(status_code=503, detail="Платная генерация пока выключена. Сначала завершаем проверку провайдера.")
    if body.kind not in ("image", "video"):
        raise HTTPException(status_code=400, detail="Неизвестный тип генерации")
    allowed = ("1K", "2K", "4K") if body.kind == "image" else ("480p", "720p")
    if body.resolution not in allowed:
        raise HTTPException(status_code=400, detail="Недопустимое качество")
    price = GENERATION_PRICES.get((body.kind, body.resolution))
    if not price:
        raise HTTPException(status_code=400, detail="Для этого качества нет тарифа")
    with db_connect() as conn:
        row = conn.execute("""
            INSERT INTO generation_history
                (max_user_id, generation_type, prompt, status, resolution, price_credits)
            VALUES (%s, %s, %s, 'draft', %s, %s)
            RETURNING id
        """, (user_id, body.kind, body.prompt, body.resolution, price)).fetchone()
        generation = {
            "id": int(row["id"]), "generation_type": body.kind, "prompt": body.prompt,
            "resolution": body.resolution, "price_credits": price,
            "status": "draft", "reference_image_url": None
        }
    launched, result = await launch_generation(user_id, generation)
    if not launched:
        raise HTTPException(status_code=409, detail=result)
    return {"ok": True, "generation_id": generation["id"], "task_id": result, "price": price}


@router.post("/miniapp/api/payment")
async def miniapp_payment(body: PaymentBody, authorization: str | None = Header(default=None)):
    user_id = _current_user(authorization)
    if os.getenv("ENABLE_PAYMENTS", "false").strip().lower() not in ("1", "true", "yes", "on"):
        raise HTTPException(status_code=503, detail="Приём платежей пока выключен до проверки кассовых и налоговых настроек.")
    if body.credits not in (100, 300, 500, 1000):
        raise HTTPException(status_code=400, detail="Неизвестный пакет")
    from main import create_pending_purchase_order, required_env
    from yookassa import create_yookassa_payment, save_payment_id
    order_code = create_pending_purchase_order(user_id, body.credits, body.credits)
    payment_id, payment_url = await create_yookassa_payment(
        order_code=order_code, amount_rub=body.credits,
        credits=body.credits, payment_method="bank_card"
    )
    save_payment_id(required_env("DATABASE_URL"), order_code, payment_id)
    return {"order_code": order_code, "payment_url": payment_url, "credits": body.credits}
