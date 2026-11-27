import os
import secrets
import ssl
from datetime import timezone

import httpx
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel, Field, HttpUrl

from gen202 import Gen202Client, Gen202Error
from yookassa import (
    create_yookassa_payment,
    save_payment_id,
    fetch_yookassa_payment,
    fetch_yookassa_refund,
    credit_paid_order,
    mark_payment_cancelled,
    process_yookassa_refund,
)

load_dotenv()

app = FastAPI(title="NEUROPIX AI bridge", version="1.1.0")
MAX_API = "https://platform-api2.max.ru"
_db_initialized = False


class ImageRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20000)
    aspect_ratio: str = "1:1"
    resolution: str = "1K"
    output_format: str = "jpg"
    image_input: list[HttpUrl] | None = None


class VideoRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=30000)
    duration: int = Field(default=5, ge=4, le=30)
    aspect_ratio: str = "9:16"
    resolution: str = "480p"
    generate_audio: bool = True
    first_frame_url: HttpUrl | None = None
    last_frame_url: HttpUrl | None = None
    reference_image_urls: list[HttpUrl] | None = None
    reference_video_urls: list[HttpUrl] | None = None
    reference_audio_urls: list[HttpUrl] | None = None


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise HTTPException(status_code=503, detail=f"Не настроена переменная {name}")
    return value


def db_connect():
    try:
        return psycopg.connect(
            required_env("DATABASE_URL"),
            connect_timeout=12,
            row_factory=dict_row,
        )
    except psycopg.Error as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Ошибка подключения к базе данных: {type(exc).__name__}",
        ) from exc


def initialize_database():
    global _db_initialized
    if _db_initialized:
        return
    with db_connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                max_user_id BIGINT PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                balance INTEGER NOT NULL DEFAULT 0 CHECK (balance >= 0),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS credit_transactions (
                id BIGSERIAL PRIMARY KEY,
                max_user_id BIGINT NOT NULL REFERENCES users(max_user_id),
                amount INTEGER NOT NULL,
                reason TEXT NOT NULL,
                idempotency_key TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS credit_transactions_idempotency_key_unique
            ON credit_transactions(idempotency_key)
            WHERE idempotency_key IS NOT NULL
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS purchase_orders (
                id BIGSERIAL PRIMARY KEY,
                order_code TEXT NOT NULL UNIQUE,
                max_user_id BIGINT NOT NULL REFERENCES users(max_user_id),
                package_credits INTEGER NOT NULL CHECK (package_credits > 0),
                amount_rub INTEGER NOT NULL CHECK (amount_rub > 0),
                status TEXT NOT NULL DEFAULT 'awaiting_payment',
                payment_provider TEXT,
                provider_payment_id TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        conn.execute("""
            ALTER TABLE purchase_orders
            ADD COLUMN IF NOT EXISTS provider_payment_id TEXT
        """)
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS purchase_orders_provider_payment_unique
            ON purchase_orders(provider_payment_id)
            WHERE provider_payment_id IS NOT NULL
        """)
        conn.execute("""
            ALTER TABLE purchase_orders
            ADD COLUMN IF NOT EXISTS provider_refund_id TEXT
        """)
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS purchase_orders_provider_refund_unique
            ON purchase_orders(provider_refund_id)
            WHERE provider_refund_id IS NOT NULL
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS purchase_orders_user_created_idx
            ON purchase_orders(max_user_id, created_at DESC)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS generation_history (
                id BIGSERIAL PRIMARY KEY,
                max_user_id BIGINT NOT NULL REFERENCES users(max_user_id),
                generation_type TEXT NOT NULL,
                prompt TEXT,
                status TEXT NOT NULL DEFAULT 'created',
                provider_task_id TEXT,
                result_url TEXT,
                reference_image_url TEXT,
                reference_image_token TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                max_user_id BIGINT PRIMARY KEY REFERENCES users(max_user_id),
                state TEXT NOT NULL DEFAULT 'idle',
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                reference_image_url TEXT,
                reference_image_token TEXT,
                prompt_prefix TEXT
            )
        """)
        conn.execute("""
            ALTER TABLE user_sessions ADD COLUMN IF NOT EXISTS reference_image_url TEXT
        """)
        conn.execute("""
            ALTER TABLE user_sessions ADD COLUMN IF NOT EXISTS reference_image_token TEXT
        """)
        conn.execute("""
            ALTER TABLE user_sessions ADD COLUMN IF NOT EXISTS prompt_prefix TEXT
        """)
        conn.execute("""
            ALTER TABLE generation_history ADD COLUMN IF NOT EXISTS reference_image_url TEXT
        """)
        conn.execute("""
            ALTER TABLE generation_history ADD COLUMN IF NOT EXISTS reference_image_token TEXT
        """)
    _db_initialized = True


def save_user(user_id: int, username=None, first_name=None):
    initialize_database()
    with db_connect() as conn:
        conn.execute("""
            INSERT INTO users(max_user_id, username, first_name)
            VALUES (%s, %s, %s)
            ON CONFLICT(max_user_id) DO UPDATE SET
                username=EXCLUDED.username,
                first_name=EXCLUDED.first_name,
                last_seen_at=NOW()
        """, (user_id, username, first_name))


def get_user_balance(user_id: int) -> int:
    initialize_database()
    with db_connect() as conn:
        row = conn.execute(
            "SELECT balance FROM users WHERE max_user_id=%s", (user_id,)
        ).fetchone()
    return int(row["balance"]) if row else 0


def get_credit_activity(user_id: int, limit: int = 5):
    initialize_database()
    with db_connect() as conn:
        return conn.execute("""
            SELECT amount, reason, created_at FROM credit_transactions
            WHERE max_user_id=%s ORDER BY created_at DESC, id DESC LIMIT %s
        """, (user_id, limit)).fetchall()


def create_pending_purchase_order(user_id: int, credits: int, amount_rub: int) -> str:
    if credits not in (100, 300, 500, 1000) or amount_rub != credits:
        raise ValueError("Неизвестный пакет кредитов")
    initialize_database()
    code = secrets.token_hex(5).upper()
    with db_connect() as conn:
        conn.execute(
            "INSERT INTO users(max_user_id) VALUES (%s) ON CONFLICT DO NOTHING",
            (user_id,),
        )
        conn.execute("""
            INSERT INTO purchase_orders
                (order_code, max_user_id, package_credits, amount_rub, status)
            VALUES (%s, %s, %s, %s, 'awaiting_payment')
        """, (code, user_id, credits, amount_rub))
    return code


def get_recent_purchase_orders(user_id: int, limit: int = 5):
    initialize_database()
    with db_connect() as conn:
        return conn.execute("""
            SELECT order_code, package_credits, amount_rub, status, created_at
            FROM purchase_orders WHERE max_user_id=%s
            ORDER BY created_at DESC, id DESC LIMIT %s
        """, (user_id, limit)).fetchall()


def get_user_history(user_id: int, limit: int = 5):
    initialize_database()
    with db_connect() as conn:
        return conn.execute("""
            SELECT generation_type, status, prompt,
                   (reference_image_url IS NOT NULL) AS has_reference, created_at
            FROM generation_history WHERE max_user_id=%s
            ORDER BY created_at DESC LIMIT %s
        """, (user_id, limit)).fetchall()


def set_user_state(user_id: int, state: str):
    initialize_database()
    with db_connect() as conn:
        conn.execute("""
            INSERT INTO user_sessions(max_user_id, state) VALUES (%s, %s)
            ON CONFLICT(max_user_id) DO UPDATE SET
                state=EXCLUDED.state, updated_at=NOW()
        """, (user_id, state))


def get_user_state(user_id: int) -> str:
    initialize_database()
    with db_connect() as conn:
        row = conn.execute(
            "SELECT state FROM user_sessions WHERE max_user_id=%s", (user_id,)
        ).fetchone()
    return row["state"] if row else "idle"


def set_prompt_prefix(user_id: int, prefix: str | None):
    initialize_database()
    with db_connect() as conn:
        conn.execute("""
            INSERT INTO user_sessions(max_user_id, state, prompt_prefix)
            VALUES (%s, 'idle', %s)
            ON CONFLICT(max_user_id) DO UPDATE SET
                prompt_prefix=EXCLUDED.prompt_prefix, updated_at=NOW()
        """, (user_id, prefix))


def get_prompt_prefix(user_id: int) -> str:
    initialize_database()
    with db_connect() as conn:
        row = conn.execute(
            "SELECT prompt_prefix FROM user_sessions WHERE max_user_id=%s",
            (user_id,),
        ).fetchone()
    return (row["prompt_prefix"] or "") if row else ""


def save_photo_reference(user_id: int, image_url: str, image_token: str | None):
    initialize_database()
    with db_connect() as conn:
        conn.execute("""
            INSERT INTO user_sessions
                (max_user_id, state, reference_image_url, reference_image_token)
            VALUES (%s, 'waiting_photo_prompt', %s, %s)
            ON CONFLICT(max_user_id) DO UPDATE SET
                state='waiting_photo_prompt',
                reference_image_url=EXCLUDED.reference_image_url,
                reference_image_token=EXCLUDED.reference_image_token,
                updated_at=NOW()
        """, (user_id, image_url, image_token))


def get_photo_reference(user_id: int):
    initialize_database()
    with db_connect() as conn:
        row = conn.execute("""
            SELECT reference_image_url, reference_image_token
            FROM user_sessions WHERE max_user_id=%s
        """, (user_id,)).fetchone()
    return (row["reference_image_url"], row["reference_image_token"]) if row else (None, None)


def save_photo_draft(user_id: int, prompt: str) -> int:
    initialize_database()
    image_url, image_token = get_photo_reference(user_id)
    with db_connect() as conn:
        row = conn.execute("""
            INSERT INTO generation_history
                (max_user_id, generation_type, prompt, status,
                 reference_image_url, reference_image_token)
            VALUES (%s, 'image', %s, 'draft', %s, %s)
            RETURNING id
        """, (user_id, prompt, image_url, image_token)).fetchone()
    set_user_state(user_id, "waiting_photo_confirmation")
    return int(row["id"])


def save_video_draft(user_id: int, prompt: str) -> int:
    initialize_database()
    with db_connect() as conn:
        row = conn.execute("""
            INSERT INTO generation_history(max_user_id, generation_type, prompt, status)
            VALUES (%s, 'video', %s, 'draft') RETURNING id
        """, (user_id, prompt)).fetchone()
    set_user_state(user_id, "waiting_video_confirmation")
    return int(row["id"])


def confirm_latest_draft(user_id: int):
    initialize_database()
    with db_connect() as conn:
        row = conn.execute("""
            SELECT id, generation_type FROM generation_history
            WHERE max_user_id=%s AND generation_type IN ('image','video')
              AND status='draft'
            ORDER BY created_at DESC, id DESC LIMIT 1
        """, (user_id,)).fetchone()
        if not row:
            return False, None
        conn.execute("""
            UPDATE generation_history SET status='ready_for_generation'
            WHERE id=%s AND max_user_id=%s
        """, (row["id"], user_id))
        if row["generation_type"] == "image":
            conn.execute("""
                UPDATE user_sessions
                SET reference_image_url=NULL, reference_image_token=NULL
                WHERE max_user_id=%s
            """, (user_id,))
    set_user_state(user_id, "idle")
    return True, row["generation_type"]


def cancel_latest_draft(user_id: int):
    initialize_database()
    with db_connect() as conn:
        conn.execute("""
            UPDATE generation_history SET status='cancelled'
            WHERE id=(
                SELECT id FROM generation_history
                WHERE max_user_id=%s AND status='draft'
                ORDER BY created_at DESC, id DESC LIMIT 1
            )
        """, (user_id,))
        conn.execute("""
            UPDATE user_sessions SET reference_image_url=NULL,
                reference_image_token=NULL, prompt_prefix=NULL
            WHERE max_user_id=%s
        """, (user_id,))
    set_user_state(user_id, "idle")


def max_ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    ca_file = os.getenv("MAX_CA_FILE", "").strip()
    if ca_file:
        if not os.path.isfile(ca_file):
            raise HTTPException(status_code=500, detail="Файл MAX_CA_FILE не найден")
        context.load_verify_locations(cafile=ca_file)
    return context


def max_headers():
    return {
        "Authorization": required_env("MAX_BOT_TOKEN"),
        "Content-Type": "application/json",
    }


def message_keyboard(rows: list[list[str]]):
    return [[{"type": "message", "text": label} for label in row] for row in rows]


def main_keyboard():
    return message_keyboard([
        ["Создать фото", "Создать видео"],
        ["Каталог эффектов"],
        ["Тарифы", "Мой баланс"],
        ["Купить кредиты", "Мои заказы"],
        ["История", "Поддержка"],
    ])


def credit_packages_keyboard():
    return message_keyboard([
        ["100 кредитов — 100 ₽", "300 кредитов — 300 ₽"],
        ["500 кредитов — 500 ₽", "1 000 кредитов — 1 000 ₽"],
        ["Назад в меню"],
    ])


def effects_keyboard():
    return message_keyboard([
        ["Кинематографичный портрет", "Замена фона"],
        ["Рекламное фото товара", "Фото в стиле ретро"],
        ["Автомобильное видео", "Рекламный видеоролик"],
        ["Назад в меню"],
    ])


def confirmation_keyboard():
    return message_keyboard([
        ["Подтвердить запрос"],
        ["Изменить описание", "Отменить"],
    ])


async def send_max(user_id: int, text: str, *, with_menu=False, keyboard=None):
    selected = keyboard if keyboard is not None else (main_keyboard() if with_menu else None)
    payload = {"text": text[:4000]}
    if selected:
        payload["attachments"] = [{
            "type": "inline_keyboard",
            "payload": {"buttons": selected},
        }]
    try:
        async with httpx.AsyncClient(timeout=20, verify=max_ssl_context()) as client:
            response = await client.post(
                f"{MAX_API}/messages",
                params={"user_id": user_id},
                headers=max_headers(),
                json=payload,
            )
        if response.is_error:
            raise HTTPException(
                status_code=502,
                detail=f"MAX API вернул HTTP {response.status_code}",
            )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ошибка соединения с MAX API: {type(exc).__name__}",
        ) from exc


@app.get("/health")
async def health():
    status = "not_configured"
    if os.getenv("DATABASE_URL", "").strip():
        try:
            initialize_database()
            with db_connect() as conn:
                conn.execute("SELECT 1")
            status = "connected"
        except Exception:
            status = "error"
    return {
        "ok": True,
        "service": "NEUROPIX AI bridge",
        "version": app.version,
        "database": status,
    }


@app.post("/generate/image")
async def generate_image(body: ImageRequest):
    try:
        return await Gen202Client().create_image(
            prompt=body.prompt,
            aspect_ratio=body.aspect_ratio,
            resolution=body.resolution,
            output_format=body.output_format,
            image_input=[str(x) for x in body.image_input] if body.image_input else None,
        )
    except Gen202Error as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/generate/video")
async def generate_video(body: VideoRequest):
    try:
        return await Gen202Client().create_video(
            prompt=body.prompt,
            duration=body.duration,
            aspect_ratio=body.aspect_ratio,
            resolution=body.resolution,
            generate_audio=body.generate_audio,
            first_frame_url=str(body.first_frame_url) if body.first_frame_url else None,
            last_frame_url=str(body.last_frame_url) if body.last_frame_url else None,
            reference_image_urls=[str(x) for x in body.reference_image_urls] if body.reference_image_urls else None,
            reference_video_urls=[str(x) for x in body.reference_video_urls] if body.reference_video_urls else None,
            reference_audio_urls=[str(x) for x in body.reference_audio_urls] if body.reference_audio_urls else None,
        )
    except Gen202Error as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/tasks/{task_id}")
async def task_status(task_id: str):
    try:
        return await Gen202Client().get_task(task_id)
    except Gen202Error as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/payment/return")
async def payment_return():
    return {"message": "Возврат после оплаты. Проверь баланс кредитов в NEUROPIX."}


@app.post("/webhook/yookassa")
async def yookassa_webhook(request: Request):
    notification = await request.json()
    event = notification.get("event")
    obj = notification.get("object") or {}
    database_url = required_env("DATABASE_URL")

    if event in ("payment.succeeded", "payment.canceled"):
        payment_id = obj.get("id")
        if not payment_id:
            return {"ok": True}
        payment = await fetch_yookassa_payment(str(payment_id))
        if event == "payment.succeeded":
            credited = credit_paid_order(database_url, payment)
            if credited:
                order_code = (payment.get("metadata") or {}).get("order_code")
                with db_connect() as conn:
                    order = conn.execute(
                        "SELECT max_user_id, package_credits FROM purchase_orders WHERE order_code=%s",
                        (order_code,),
                    ).fetchone()
                if order:
                    await send_max(
                        int(order["max_user_id"]),
                        f"Оплата получена! Начислено {order['package_credits']} кредитов.",
                        with_menu=True,
                    )
        else:
            mark_payment_cancelled(database_url, payment)
        return {"ok": True}

    if event == "refund.succeeded":
        refund_id = obj.get("id")
        if not refund_id:
            return {"ok": True}
        refund = await fetch_yookassa_refund(str(refund_id))
        result = process_yookassa_refund(database_url, refund)
        return {"ok": True, "refund_result": result}

    return {"ok": True}


@app.post("/webhook/max")
async def max_webhook(
    request: Request,
    x_max_bot_api_secret: str | None = Header(default=None),
):
    if x_max_bot_api_secret != required_env("MAX_WEBHOOK_SECRET"):
        raise HTTPException(status_code=401, detail="Неверный секрет webhook")

    update = await request.json()
    update_type = update.get("update_type")

    if update_type == "bot_started":
        user = update.get("user") or {}
        user_id = user.get("user_id")
        if user_id:
            save_user(int(user_id), user.get("username"), user.get("first_name"))
            await send_max(
                int(user_id),
                "Добро пожаловать в NEUROPIX!\nВыбери, что хочешь создать:",
                with_menu=True,
            )
        return {"ok": True}

    if update_type != "message_created":
        return {"ok": True}

    message = update.get("message") or {}
    sender = message.get("sender") or {}
    user_id = sender.get("user_id")
    body = message.get("body") or {}
    message_text = (body.get("text") or message.get("text") or "").strip()
    if not user_id:
        return {"ok": True}

    user_id = int(user_id)
    save_user(user_id, sender.get("username"), sender.get("first_name"))

    attachments = body.get("attachments") or message.get("attachments") or []
    image = next((x for x in attachments if x.get("type") == "image"), None)
    if image:
        payload = image.get("payload") or {}
        image_url = payload.get("url")
        if not image_url:
            await send_max(user_id, "Не удалось получить ссылку на фото. Отправь его ещё раз.", with_menu=True)
            return {"ok": True}
        save_photo_reference(user_id, str(image_url), str(payload.get("token")) if payload.get("token") else None)
        await send_max(
            user_id,
            "Фотография получена. Теперь напиши, что нужно изменить: фон, одежду, освещение или другое. Генерация пока выключена.",
            keyboard=message_keyboard([["Отменить"]]),
        )
        return {"ok": True}

    if not message_text:
        return {"ok": True}

    command = message_text.lower()
    state = get_user_state(user_id)
    show_menu, keyboard = True, None

    if command in ("/start", "старт", "начать", "меню", "главное меню"):
        set_user_state(user_id, "idle")
        answer = "NEUROPIX — создание ИИ-фото и видео. Выбери нужный раздел:"
    elif command in ("/help", "помощь"):
        answer = "Выбери раздел кнопками ниже. Подготовка запросов доступна. Генерация включается отдельно."
    elif command in ("создать фото", "фото"):
        set_prompt_prefix(user_id, None)
        set_user_state(user_id, "waiting_photo_prompt")
        answer = "Отправь фото-референс или сразу напиши описание изображения. Я подготовлю черновик. Генерация пока выключена."
        show_menu, keyboard = False, message_keyboard([["Отменить"]])
    elif command in ("создать видео", "видео"):
        set_prompt_prefix(user_id, None)
        set_user_state(user_id, "waiting_video_prompt")
        answer = "Опиши видео: что происходит, движение камеры, персонажи, окружение, стиль и длительность. Пока сохраню только черновик."
        show_menu, keyboard = False, message_keyboard([["Отменить"]])
    elif command == "каталог эффектов":
        answer = "Выбери готовый эффект, затем добавь свои детали."
        show_menu, keyboard = False, effects_keyboard()
    elif command == "назад в меню":
        set_user_state(user_id, "idle")
        answer = "Главное меню NEUROPIX:"
    elif command == "кинематографичный портрет":
        set_prompt_prefix(user_id, "Кинематографичный фотореалистичный портрет, профессиональный свет, естественная кожа. Детали: ")
        set_user_state(user_id, "waiting_photo_prompt")
        answer = "Эффект выбран. Опиши человека, одежду, фон и настроение."
        show_menu, keyboard = False, message_keyboard([["Отменить"]])
    elif command == "замена фона":
        set_prompt_prefix(user_id, "Фотореалистичная замена фона. Сохранить внешность человека, перспективу, свет и тени. Новый фон: ")
        set_user_state(user_id, "waiting_photo_prompt")
        answer = "Опиши новый фон и атмосферу. При желании прикрепи фото-референс."
        show_menu, keyboard = False, message_keyboard([["Отменить"]])
    elif command == "рекламное фото товара":
        set_prompt_prefix(user_id, "Профессиональная рекламная предметная фотография, студийное освещение. Товар и пожелания: ")
        set_user_state(user_id, "waiting_photo_prompt")
        answer = "Опиши товар, фон и стиль рекламы."
        show_menu, keyboard = False, message_keyboard([["Отменить"]])
    elif command == "фото в стиле ретро":
        set_prompt_prefix(user_id, "Ретро-фотография, аналоговая фактура, плёночное зерно. Сюжет и эпоха: ")
        set_user_state(user_id, "waiting_photo_prompt")
        answer = "Опиши сюжет и эпоху."
        show_menu, keyboard = False, message_keyboard([["Отменить"]])
    elif command == "автомобильное видео":
        set_prompt_prefix(user_id, "Кинематографичный автомобильный ролик, плавная камера, реалистичное движение. Автомобиль и сцена: ")
        set_user_state(user_id, "waiting_video_prompt")
        answer = "Опиши машину, локацию, движение и длительность."
        show_menu, keyboard = False, message_keyboard([["Отменить"]])
    elif command == "рекламный видеоролик":
        set_prompt_prefix(user_id, "Профессиональный рекламный видеоролик, плавная камера. Продукт и сценарий: ")
        set_user_state(user_id, "waiting_video_prompt")
        answer = "Опиши продукт, аудиторию, идею и сцену."
        show_menu, keyboard = False, message_keyboard([["Отменить"]])
    elif command in ("тарифы", "цены", "стоимость"):
        answer = (
            "ТАРИФЫ NEUROPIX\n\nФОТОГРАФИИ\n"
            "• 1K — 15 кредитов\n• 2K — 20 кредитов\n• 4K — 35 кредитов\n\n"
            "ВИДЕО — 5 СЕКУНД\n• 480p — 149 кредитов\n• 720p — 299 кредитов\n"
            "• 1080p — временно недоступно\n\n1 кредит = 1 рубль. "
            "Генерация пока отключена."
        )
    elif command == "купить кредиты":
        answer = (
            "ВЫБЕРИ ПАКЕТ КРЕДИТОВ\n\n"
            "• 100 кредитов — 100 ₽\n• 300 кредитов — 300 ₽\n"
            "• 500 кредитов — 500 ₽\n• 1 000 кредитов — 1 000 ₽"
        )
        show_menu, keyboard = False, credit_packages_keyboard()
    elif command in ("100 кредитов — 100 ₽", "300 кредитов — 300 ₽", "500 кредитов — 500 ₽", "1 000 кредитов — 1 000 ₽"):
        packages = {
            "100 кредитов — 100 ₽": (100, 100),
            "300 кредитов — 300 ₽": (300, 300),
            "500 кредитов — 500 ₽": (500, 500),
            "1 000 кредитов — 1 000 ₽": (1000, 1000),
        }
        credits, amount = packages[command]
        order_code = create_pending_purchase_order(user_id, credits, amount)
        payment_id, payment_url = await create_yookassa_payment(
            order_code=order_code,
            amount_rub=amount,
            credits=credits,
        )
        save_payment_id(required_env("DATABASE_URL"), order_code, payment_id)
        answer = (
            f"Заказ №{order_code}\nПакет: {credits} кредитов — {amount} ₽.\n\n"
            f"Перейди по ссылке для оплаты:\n{payment_url}\n\n"
            "Кредиты будут начислены после подтверждения платежа."
        )
    elif command == "мои заказы":
        orders = get_recent_purchase_orders(user_id)
        if orders:
            names = {
                "awaiting_payment": "ожидает оплаты",
                "paid": "оплачен",
                "cancelled": "отменён",
                "failed": "ошибка оплаты",
                "refunded": "возврат выполнен",
                "refund_review": "возврат — нужна проверка",
            }
            answer = "ТВОИ ЗАКАЗЫ\n\n" + "\n".join(
                f"• {o['order_code']} — {o['package_credits']} кредитов, "
                f"{o['amount_rub']} ₽; {names.get(o['status'], o['status'])}"
                for o in orders
            )
        else:
            answer = "Заказов пока нет."
    elif command == "мой баланс":
        balance = get_user_balance(user_id)
        activity = get_credit_activity(user_id)
        lines = [
            f"• {'+' if int(x['amount']) > 0 else ''}{x['amount']} — {x['reason']}"
            for x in activity
        ]
        answer = f"Твой баланс: {balance} кредитов."
        if lines:
            answer += "\nПоследние операции:\n" + "\n".join(lines)
    elif command == "история":
        history = get_user_history(user_id)
        if not history:
            answer = "История пока пустая."
        else:
            answer = "Последние операции:\n" + "\n".join(
                f"• {'фото' if x['generation_type']=='image' else 'видео'} — {x['status']}"
                + (" · с референсом" if x["has_reference"] else "")
                + f"\n  {(x['prompt'] or '')[:100]}"
                for x in history
            )
    elif command == "поддержка":
        answer = "Контакт поддержки будет добавлен перед публичным запуском."
    elif command in ("отменить", "отмена"):
        cancel_latest_draft(user_id)
        answer = "Подготовка запроса отменена. Выбери следующий раздел:"
    elif command in ("изменить описание", "изменить"):
        with db_connect() as conn:
            pending = conn.execute("""
                SELECT generation_type FROM generation_history
                WHERE max_user_id=%s AND status='draft'
                ORDER BY created_at DESC, id DESC LIMIT 1
            """, (user_id,)).fetchone()
        if pending and pending["generation_type"] == "video":
            set_user_state(user_id, "waiting_video_prompt")
            answer = "Напиши новое описание видео."
        else:
            set_user_state(user_id, "waiting_photo_prompt")
            answer = "Напиши новое описание фото."
        show_menu, keyboard = False, message_keyboard([["Отменить"]])
    elif command in ("подтвердить запрос", "подтвердить"):
        confirmed, kind = confirm_latest_draft(user_id)
        answer = (
            f"Черновик {'фото' if kind == 'image' else 'видео'} сохранён. "
            "Генерация пока отключена, кредиты не списаны."
            if confirmed else "Нет подготовленного запроса для подтверждения."
        )
    elif state == "waiting_photo_prompt":
        if len(message_text) < 5:
            answer = "Добавь деталей — описание слишком короткое."
            show_menu, keyboard = False, message_keyboard([["Отменить"]])
        else:
            prompt = (get_prompt_prefix(user_id) + message_text).strip()[:20000]
            draft_id = save_photo_draft(user_id, prompt)
            set_prompt_prefix(user_id, None)
            ref_url, _ = get_photo_reference(user_id)
            answer = (
                f"Черновик запроса №{draft_id}:\n\n"
                f"{'Фотография-референс прикреплена.' if ref_url else 'Фотография-референс не прикреплена.'}\n\n"
                f"{prompt[:1200]}\n\n"
                "Подтверждение сохранит запрос. Генерация пока не запускается."
            )
            show_menu, keyboard = False, confirmation_keyboard()
    elif state == "waiting_video_prompt":
        if len(message_text) < 5:
            answer = "Добавь деталей: что происходит, как движется камера и сколько длится сцена."
            show_menu, keyboard = False, message_keyboard([["Отменить"]])
        else:
            prompt = (get_prompt_prefix(user_id) + message_text).strip()[:30000]
            draft_id = save_video_draft(user_id, prompt)
            set_prompt_prefix(user_id, None)
            answer = (
                f"Черновик видео №{draft_id}:\n\n{prompt[:1200]}\n\n"
                "Подтверждение сохранит запрос. Генерация пока не запускается."
            )
            show_menu, keyboard = False, confirmation_keyboard()
    elif state in ("waiting_photo_confirmation", "waiting_video_confirmation"):
        answer = "Сначала подтверди подготовленный запрос, измени описание или отмени его."
        show_menu, keyboard = False, confirmation_keyboard()
    else:
        answer = "Не понял команду. Выбери раздел кнопками меню ниже."

    await send_max(user_id, answer, with_menu=show_menu, keyboard=keyboard)
    return {"ok": True}


@app.post("/max/register-webhook")
async def register_webhook(x_admin_secret: str | None = Header(default=None)):
    if x_admin_secret != required_env("MAX_ADMIN_SECRET"):
        raise HTTPException(status_code=401, detail="Неверный admin secret")
    webhook_url = required_env("MAX_WEBHOOK_URL")
    if not webhook_url.startswith("https://"):
        raise HTTPException(status_code=500, detail="MAX_WEBHOOK_URL должен начинаться с https://")
    payload = {
        "url": webhook_url,
        "update_types": ["message_created", "bot_started"],
        "secret": required_env("MAX_WEBHOOK_SECRET"),
    }
    try:
        async with httpx.AsyncClient(timeout=20, verify=max_ssl_context()) as client:
            response = await client.post(
                f"{MAX_API}/subscriptions",
                headers=max_headers(),
                json=payload,
            )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Ошибка соединения с MAX: {type(exc).__name__}") from exc
    try:
        data = response.json()
    except ValueError:
        data = {"response": response.text[:300]}
    if response.is_error or (isinstance(data, dict) and data.get("success") is False):
        raise HTTPException(status_code=502, detail={"status": response.status_code, "response": data})
    return {"ok": True, "response": data}
