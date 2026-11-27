import os
import ssl
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel, Field, HttpUrl
import httpx
import psycopg
from psycopg.rows import dict_row

from gen202 import Gen202Client, Gen202Error

load_dotenv()

app = FastAPI(title="NEUROPIX AI bridge", version="0.4.0")

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


def gen_client() -> Gen202Client:
    try:
        return Gen202Client()
    except Gen202Error as exc:
        raise HTTPException(status_code=503, detail=str(exc))


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise HTTPException(
            status_code=503,
            detail=f"Environment variable {name} is not configured",
        )
    return value


def db_connect():
    database_url = required_env("DATABASE_URL")
    try:
        return psycopg.connect(
            database_url,
            connect_timeout=12,
            row_factory=dict_row,
        )
    except psycopg.Error as exc:
        # Do not return the connection string or credentials to the caller.
        raise HTTPException(
            status_code=503,
            detail=f"Database connection failed: {type(exc).__name__}",
        ) from exc


def initialize_database() -> None:
    global _db_initialized
    if _db_initialized:
        return

    try:
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
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
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
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
        _db_initialized = True
    except HTTPException:
        raise
    except psycopg.Error as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Database initialization failed: {type(exc).__name__}",
        ) from exc


def save_user(user_id: int, username: str | None, first_name: str | None) -> None:
    initialize_database()
    try:
        with db_connect() as conn:
            conn.execute(
                """
                INSERT INTO users (max_user_id, username, first_name)
                VALUES (%s, %s, %s)
                ON CONFLICT (max_user_id) DO UPDATE SET
                    username = EXCLUDED.username,
                    first_name = EXCLUDED.first_name,
                    last_seen_at = NOW()
                """,
                (user_id, username, first_name),
            )
    except HTTPException:
        raise
    except psycopg.Error as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not save user: {type(exc).__name__}",
        ) from exc


def get_user_balance(user_id: int) -> int:
    initialize_database()
    try:
        with db_connect() as conn:
            row = conn.execute(
                "SELECT balance FROM users WHERE max_user_id = %s",
                (user_id,),
            ).fetchone()
        return int(row["balance"]) if row else 0
    except HTTPException:
        raise
    except psycopg.Error as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not read balance: {type(exc).__name__}",
        ) from exc


def get_user_history(user_id: int, limit: int = 5) -> list[dict]:
    initialize_database()
    try:
        with db_connect() as conn:
            rows = conn.execute(
                """
                SELECT generation_type, status, created_at
                FROM generation_history
                WHERE max_user_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (user_id, limit),
            ).fetchall()
        return rows
    except HTTPException:
        raise
    except psycopg.Error as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not read history: {type(exc).__name__}",
        ) from exc


def max_ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    ca_file = os.getenv("MAX_CA_FILE", "").strip()
    if ca_file:
        if not os.path.isfile(ca_file):
            raise HTTPException(
                status_code=500,
                detail="MAX_CA_FILE path does not exist on the server",
            )
        try:
            context.load_verify_locations(cafile=ca_file)
        except (OSError, ssl.SSLError) as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not load MAX CA certificate: {type(exc).__name__}",
            ) from exc
    return context


def max_headers() -> dict[str, str]:
    return {
        "Authorization": required_env("MAX_BOT_TOKEN"),
        "Content-Type": "application/json",
    }


def main_keyboard() -> list[list[dict]]:
    return [
        [{"type": "message", "text": "Создать фото"},
         {"type": "message", "text": "Создать видео"}],
        [{"type": "message", "text": "Каталог эффектов"}],
        [{"type": "message", "text": "Мой баланс"},
         {"type": "message", "text": "Купить кредиты"}],
        [{"type": "message", "text": "История"},
         {"type": "message", "text": "Поддержка"}],
    ]


async def send_max(user_id: int, text: str, *, with_menu: bool = False) -> None:
    payload = {"text": text[:4000]}
    if with_menu:
        payload["attachments"] = [{
            "type": "inline_keyboard",
            "payload": {"buttons": main_keyboard()},
        }]

    try:
        async with httpx.AsyncClient(timeout=20, verify=max_ssl_context()) as http:
            response = await http.post(
                f"{MAX_API}/messages",
                params={"user_id": user_id},
                headers=max_headers(),
                json=payload,
            )
        if response.is_error:
            raise HTTPException(
                status_code=502,
                detail=f"MAX API returned HTTP {response.status_code}",
            )
    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not connect securely to MAX API: {type(exc).__name__}",
        ) from exc


@app.get("/health")
async def health():
    database_status = "not_configured"
    if os.getenv("DATABASE_URL", "").strip():
        try:
            initialize_database()
            with db_connect() as conn:
                conn.execute("SELECT 1")
            database_status = "connected"
        except HTTPException:
            database_status = "error"
    return {
        "ok": True,
        "service": "NEUROPIX AI bridge",
        "database": database_status,
    }


@app.post("/generate/image")
async def generate_image(body: ImageRequest):
    # Generation endpoint is retained, but the MAX chat flow does not call it yet.
    try:
        return await gen_client().create_image(
            prompt=body.prompt,
            aspect_ratio=body.aspect_ratio,
            resolution=body.resolution,
            output_format=body.output_format,
            image_input=[str(x) for x in body.image_input] if body.image_input else None,
        )
    except Gen202Error as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/generate/video")
async def generate_video(body: VideoRequest):
    try:
        return await gen_client().create_video(
            prompt=body.prompt,
            duration=body.duration,
            aspect_ratio=body.aspect_ratio,
            resolution=body.resolution,
            generate_audio=body.generate_audio,
            first_frame_url=str(body.first_frame_url) if body.first_frame_url else None,
            last_frame_url=str(body.last_frame_url) if body.last_frame_url else None,
            reference_image_urls=[str(x) for x in body.reference_image_urls]
            if body.reference_image_urls else None,
            reference_video_urls=[str(x) for x in body.reference_video_urls]
            if body.reference_video_urls else None,
            reference_audio_urls=[str(x) for x in body.reference_audio_urls]
            if body.reference_audio_urls else None,
        )
    except Gen202Error as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/tasks/{task_id}")
async def task_status(task_id: str):
    try:
        return await gen_client().get_task(task_id)
    except Gen202Error as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/webhook/max")
async def max_webhook(
    request: Request,
    x_max_bot_api_secret: str | None = Header(default=None),
):
    if x_max_bot_api_secret != required_env("MAX_WEBHOOK_SECRET"):
        raise HTTPException(status_code=401, detail="Invalid webhook secret")

    update = await request.json()
    update_type = update.get("update_type")

    if update_type == "bot_started":
        user = update.get("user") or {}
        user_id = user.get("user_id")
        if user_id:
            save_user(
                int(user_id),
                user.get("username"),
                user.get("first_name"),
            )
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

    save_user(
        int(user_id),
        sender.get("username"),
        sender.get("first_name"),
    )

    if not message_text:
        return {"ok": True}

    command = message_text.lower()

    if command in ("/start", "старт", "начать", "меню", "главное меню"):
        answer = "NEUROPIX — создание ИИ-фото и видео. Выбери нужный раздел:"
        show_menu = True
    elif command in ("/help", "помощь"):
        answer = (
            "Выбери раздел кнопками ниже.\n"
            "Генерация и оплата пока не включены."
        )
        show_menu = True
    elif command in ("создать фото", "фото"):
        answer = (
            "Создание фото выбрано. Скоро подключим загрузку референса, "
            "промпт и Nano Banana 2.1. Сейчас запуск отключён — кредиты не расходуются."
        )
        show_menu = True
    elif command in ("создать видео", "видео"):
        answer = (
            "Создание видео выбрано. Скоро подключим промпт, референсы "
            "и Seedance 2.5. Сейчас запуск отключён — кредиты не расходуются."
        )
        show_menu = True
    elif command == "каталог эффектов":
        answer = (
            "Каталог эффектов готовится: кинематографичный портрет, смена образа, "
            "оживление фото, рекламный ролик и другие шаблоны."
        )
        show_menu = True
    elif command == "мой баланс":
        balance = get_user_balance(int(user_id))
        answer = f"Твой баланс: {balance} кредитов.\nПокупка и начисление кредитов пока не включены."
        show_menu = True
    elif command == "купить кредиты":
        answer = (
            "Покупка кредитов пока не подключена. Сначала настроим платёжный "
            "сервис и безопасное автоматическое начисление."
        )
        show_menu = True
    elif command == "история":
        history = get_user_history(int(user_id))
        if not history:
            answer = "История пока пустая. Здесь будут отображаться твои генерации."
        else:
            lines = [
                f"• {item['generation_type']} — {item['status']} "
                f"({item['created_at'].astimezone(timezone.utc).strftime('%d.%m.%Y %H:%M UTC')})"
                for item in history
            ]
            answer = "Последние операции:\n" + "\n".join(lines)
        show_menu = True
    elif command == "поддержка":
        answer = "По вопросам работы NEUROPIX напиши администратору: контакт поддержки будет добавлен перед запуском."
        show_menu = True
    elif command.startswith(("фото:", "image:")):
        answer = "Запрос принят, но генерация пока отключена. Кредиты не расходуются."
        show_menu = True
    elif command.startswith(("видео:", "video:")):
        answer = "Запрос принят, но генерация пока отключена. Кредиты не расходуются."
        show_menu = True
    else:
        answer = "Не понял команду. Выбери раздел кнопками меню ниже."
        show_menu = True

    await send_max(int(user_id), answer, with_menu=show_menu)
    return {"ok": True}


@app.post("/max/register-webhook")
async def register_webhook(x_admin_secret: str | None = Header(default=None)):
    if x_admin_secret != required_env("MAX_ADMIN_SECRET"):
        raise HTTPException(status_code=401, detail="Invalid admin secret")

    webhook_url = required_env("MAX_WEBHOOK_URL")
    if not webhook_url.startswith("https://"):
        raise HTTPException(status_code=500, detail="MAX_WEBHOOK_URL must use HTTPS")

    payload = {
        "url": webhook_url,
        "update_types": ["message_created", "bot_started"],
        "secret": required_env("MAX_WEBHOOK_SECRET"),
    }

    try:
        async with httpx.AsyncClient(timeout=20, verify=max_ssl_context()) as http:
            response = await http.post(
                f"{MAX_API}/subscriptions",
                headers=max_headers(),
                json=payload,
            )
    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not connect securely to MAX API: {type(exc).__name__}",
        ) from exc

    try:
        data = response.json()
    except ValueError:
        data = {"raw": response.text[:400]}

    if response.is_error or (
        isinstance(data, dict) and data.get("success") is False
    ):
        raise HTTPException(
            status_code=502,
            detail={"status": response.status_code, "response": data},
        )

    return {"ok": True, "response": data}
