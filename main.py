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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    max_user_id BIGINT PRIMARY KEY REFERENCES users(max_user_id),
                    state TEXT NOT NULL DEFAULT 'idle',
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
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


def set_user_state(user_id: int, state: str) -> None:
    initialize_database()
    with db_connect() as conn:
        conn.execute(
            """
            INSERT INTO user_sessions (max_user_id, state)
            VALUES (%s, %s)
            ON CONFLICT (max_user_id) DO UPDATE SET
                state = EXCLUDED.state,
                updated_at = NOW()
            """,
            (user_id, state),
        )


def get_user_state(user_id: int) -> str:
    initialize_database()
    with db_connect() as conn:
        row = conn.execute(
            "SELECT state FROM user_sessions WHERE max_user_id = %s",
            (user_id,),
        ).fetchone()
    return row["state"] if row else "idle"


def save_photo_draft(user_id: int, prompt: str) -> int:
    initialize_database()
    with db_connect() as conn:
        row = conn.execute(
            """
            INSERT INTO generation_history
                (max_user_id, generation_type, prompt, status)
            VALUES (%s, 'image', %s, 'draft')
            RETURNING id
            """,
            (user_id, prompt),
        ).fetchone()
    set_user_state(user_id, "waiting_photo_confirmation")
    return int(row["id"])


def confirm_latest_photo_draft(user_id: int) -> bool:
    initialize_database()
    with db_connect() as conn:
        row = conn.execute(
            """
            SELECT id FROM generation_history
            WHERE max_user_id = %s AND generation_type = 'image' AND status = 'draft'
            ORDER BY created_at DESC LIMIT 1
            """,
            (user_id,),
        ).fetchone()
        if not row:
            return False
        conn.execute(
            "UPDATE generation_history SET status = 'ready_for_generation' WHERE id = %s AND max_user_id = %s",
            (row["id"], user_id),
        )
    set_user_state(user_id, "idle")
    return True


def cancel_latest_photo_draft(user_id: int) -> None:
    initialize_database()
    with db_connect() as conn:
        conn.execute(
            """
            UPDATE generation_history SET status = 'cancelled'
            WHERE id = (
                SELECT id FROM generation_history
                WHERE max_user_id = %s AND generation_type = 'image' AND status = 'draft'
                ORDER BY created_at DESC LIMIT 1
            )
            """,
            (user_id,),
        )
    set_user_state(user_id, "idle")


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


def message_keyboard(rows: list[list[str]]) -> list[list[dict]]:
    return [[{"type": "message", "text": label} for label in row] for row in rows]


def main_keyboard() -> list[list[dict]]:
    return message_keyboard([
        ["Создать фото", "Создать видео"],
        ["Каталог эффектов"],
        ["Мой баланс", "Купить кредиты"],
        ["История", "Поддержка"],
    ])


def photo_confirmation_keyboard() -> list[list[dict]]:
    return message_keyboard([
        ["Подтвердить запрос"],
        ["Изменить описание", "Отменить"],
    ])


async def send_max(
    user_id: int,
    text: str,
    *,
    with_menu: bool = False,
    keyboard: list[list[dict]] | None = None,
) -> None:
    payload = {"text": text[:4000]}
    selected_keyboard = keyboard if keyboard is not None else (main_keyboard() if with_menu else None)
    if selected_keyboard:
        payload["attachments"] = [{
            "type": "inline_keyboard",
            "payload": {"buttons": selected_keyboard},
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
    current_state = get_user_state(int(user_id))

    if command in ("/start", "старт", "начать", "меню", "главное меню"):
        set_user_state(int(user_id), "idle")
        answer = "NEUROPIX — создание ИИ-фото и видео. Выбери нужный раздел:"
        show_menu, keyboard = True, None
    elif command in ("/help", "помощь"):
        answer = "Выбери раздел кнопками ниже. Подготовка запроса на фото доступна, запуск генерации и оплата пока отключены."
        show_menu, keyboard = True, None
    elif command in ("создать фото", "фото"):
        set_user_state(int(user_id), "waiting_photo_prompt")
        answer = (
            "Опиши, какое фото хочешь получить. Напиши подробный запрос одним сообщением: "
            "кто или что должно быть на изображении, стиль, фон, одежда, освещение и формат. "
            "Пока принимаем текстовое описание; загрузку фото-референса добавим следующим этапом. "
            "Генерация не запустится и кредиты не спишутся."
        )
        show_menu, keyboard = False, message_keyboard([["Отменить"]])
    elif command in ("создать видео", "видео"):
        answer = "Создание видео пока находится в разработке. Выбери «Создать фото» или вернись в главное меню."
        show_menu, keyboard = True, None
    elif command == "каталог эффектов":
        answer = "Каталог эффектов готовится: кинематографичный портрет, смена образа, оживление фото, рекламный ролик и другие шаблоны."
        show_menu, keyboard = True, None
    elif command == "мой баланс":
        balance = get_user_balance(int(user_id))
        answer = f"Твой баланс: {balance} кредитов. Покупка и начисление кредитов пока не включены."
        show_menu, keyboard = True, None
    elif command == "купить кредиты":
        answer = "Покупка кредитов пока не подключена. Сначала настроим платёжный сервис и безопасное автоматическое начисление."
        show_menu, keyboard = True, None
    elif command == "история":
        history = get_user_history(int(user_id))
        if not history:
            answer = "История пока пустая. Здесь будут отображаться твои генерации и подготовленные запросы."
        else:
            lines = [
                f"• {item['generation_type']} — {item['status']} "
                f"({item['created_at'].astimezone(timezone.utc).strftime('%d.%m.%Y %H:%M UTC')})"
                for item in history
            ]
            answer = "Последние операции:\\n" + "\\n".join(lines)
        show_menu, keyboard = True, None
    elif command == "поддержка":
        answer = "По вопросам работы NEUROPIX контакт поддержки будет добавлен перед запуском."
        show_menu, keyboard = True, None
    elif command in ("отменить", "отмена"):
        cancel_latest_photo_draft(int(user_id))
        answer = "Подготовка запроса отменена. Выбери следующий раздел:"
        show_menu, keyboard = True, None
    elif command in ("изменить описание", "изменить"):
        set_user_state(int(user_id), "waiting_photo_prompt")
        answer = "Напиши новое описание изображения."
        show_menu, keyboard = False, message_keyboard([["Отменить"]])
    elif command in ("подтвердить запрос", "подтвердить"):
        if confirm_latest_photo_draft(int(user_id)):
            answer = (
                "Запрос сохранён и готов к следующему этапу. Сама генерация пока отключена: "
                "кредиты не списаны, задача в GEN202 не создавалась."
            )
        else:
            answer = "Нет подготовленного запроса для подтверждения."
        show_menu, keyboard = True, None
    elif current_state == "waiting_photo_prompt":
        if len(message_text) < 5:
            answer = "Добавь немного деталей — описание слишком короткое. Напиши, что именно должно быть на фото."
            show_menu, keyboard = False, message_keyboard([["Отменить"]])
        else:
            draft_id = save_photo_draft(int(user_id), message_text[:20000])
            preview = message_text if len(message_text) <= 1200 else message_text[:1200] + "…"
            answer = (
                f"Черновик запроса №{draft_id}:\\n\\n{preview}\\n\\n"
                "Проверь описание. Подтверждение только сохранит запрос для следующего этапа. "
                "Генерация пока не запускается и кредиты не расходуются."
            )
            show_menu, keyboard = False, photo_confirmation_keyboard()
    elif current_state == "waiting_photo_confirmation":
        answer = "Сначала подтверди подготовленный запрос, измени описание или отмени его."
        show_menu, keyboard = False, photo_confirmation_keyboard()
    elif command.startswith(("фото:", "image:")):
        answer = "Чтобы подготовить фото, нажми «Создать фото» и отправь описание."
        show_menu, keyboard = True, None
    elif command.startswith(("видео:", "video:")):
        answer = "Создание видео пока отключено."
        show_menu, keyboard = True, None
    else:
        answer = "Не понял команду. Выбери раздел кнопками меню ниже."
        show_menu, keyboard = True, None

    await send_max(int(user_id), answer, with_menu=show_menu, keyboard=keyboard)
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
