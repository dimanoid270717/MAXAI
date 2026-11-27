import os
import ssl
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel, Field, HttpUrl
import httpx

from gen202 import Gen202Client, Gen202Error

load_dotenv()

app = FastAPI(title="NEUROPIX AI bridge", version="0.3.0")

# MAX Bot API. SSL verification remains enabled.
MAX_API = "https://platform-api2.max.ru"


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


def max_ssl_context() -> ssl.SSLContext:
    """
    Use the normal system trust store plus the additional MAX trusted root
    certificate uploaded to Render Secret Files. Certificate verification
    and hostname verification remain enabled.
    """
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
    # MAX inline keyboard buttons of type "message" send their text as a
    # normal message, so the existing message_created webhook handles them.
    return [
        [{"type": "message", "text": "Создать фото"},
         {"type": "message", "text": "Создать видео"}],
        [{"type": "message", "text": "Каталог эффектов"}],
        [{"type": "message", "text": "Мой баланс"},
         {"type": "message", "text": "Купить кредиты"}],
        [{"type": "message", "text": "История"},
         {"type": "message", "text": "Поддержка"}],
    ]


async def send_max(
    user_id: int,
    text: str,
    *,
    with_menu: bool = False,
) -> None:
    payload = {"text": text[:4000]}
    if with_menu:
        payload["attachments"] = [{
            "type": "inline_keyboard",
            "payload": {"buttons": main_keyboard()},
        }]

    try:
        async with httpx.AsyncClient(
            timeout=20,
            verify=max_ssl_context(),
        ) as http:
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
    return {"ok": True, "service": "NEUROPIX AI bridge"}


@app.post("/generate/image")
async def generate_image(body: ImageRequest):
    try:
        return await gen_client().create_image(
            prompt=body.prompt,
            aspect_ratio=body.aspect_ratio,
            resolution=body.resolution,
            output_format=body.output_format,
            image_input=[str(x) for x in body.image_input]
            if body.image_input else None,
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
            first_frame_url=str(body.first_frame_url)
            if body.first_frame_url else None,
            last_frame_url=str(body.last_frame_url)
            if body.last_frame_url else None,
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
        user_id = (update.get("user") or {}).get("user_id")
        if user_id:
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

    if not user_id or not message_text:
        return {"ok": True}

    command = message_text.lower()

    if command in ("/start", "старт", "начать", "меню", "главное меню"):
        answer = "NEUROPIX — создание ИИ-фото и видео. Выбери нужный раздел:"
        show_menu = True
    elif command in ("/help", "помощь"):
        answer = (
            "Выбери раздел кнопками ниже.\n"
            "Пока меню работает в тестовом режиме: генерации, баланс и оплата "
            "ещё не подключены."
        )
        show_menu = True
    elif command in ("создать фото", "фото"):
        answer = (
            "Создание фото выбрано. Следующим этапом подключим загрузку "
            "референса, ввод промпта и запуск Nano Banana 2.1. "
            "Сейчас генерация отключена, кредиты не расходуются."
        )
        show_menu = True
    elif command in ("создать видео", "видео"):
        answer = (
            "Создание видео выбрано. Следующим этапом подключим промпт, "
            "референсы и параметры Seedance 2.5. "
            "Сейчас генерация отключена, кредиты не расходуются."
        )
        show_menu = True
    elif command == "каталог эффектов":
        answer = (
            "Каталог эффектов подготовим отдельно: кинематографичный портрет, "
            "смена образа, оживление фото, рекламный ролик и другие шаблоны. "
            "Пока это демонстрационный раздел."
        )
        show_menu = True
    elif command == "мой баланс":
        answer = "Баланс пока не подключён. Система учёта кредитов будет добавлена до запуска платных генераций."
        show_menu = True
    elif command == "купить кредиты":
        answer = "Покупка кредитов пока не подключена. Сначала настроим платёжный сервис и автоматическое начисление."
        show_menu = True
    elif command == "история":
        answer = "История генераций появится после подключения базы данных и самого процесса генерации."
        show_menu = True
    elif command == "поддержка":
        answer = "Раздел поддержки готовится. Здесь появится способ связаться с администратором."
        show_menu = True
    elif command.startswith(("фото:", "image:")):
        answer = (
            "Запрос получил. Генерация пока отключена, поэтому кредиты не расходуются. "
            "Скоро добавим полноценную форму создания фото."
        )
        show_menu = True
    elif command.startswith(("видео:", "video:")):
        answer = (
            "Запрос получил. Генерация пока отключена, поэтому кредиты не расходуются. "
            "Скоро добавим полноценную форму создания видео."
        )
        show_menu = True
    else:
        answer = "Не понял команду. Выбери раздел кнопками меню ниже."
        show_menu = True

    await send_max(int(user_id), answer, with_menu=show_menu)
    return {"ok": True}


@app.post("/max/register-webhook")
async def register_webhook(
    x_admin_secret: str | None = Header(default=None),
):
    if x_admin_secret != required_env("MAX_ADMIN_SECRET"):
        raise HTTPException(status_code=401, detail="Invalid admin secret")

    webhook_url = required_env("MAX_WEBHOOK_URL")
    if not webhook_url.startswith("https://"):
        raise HTTPException(
            status_code=500,
            detail="MAX_WEBHOOK_URL must use HTTPS",
        )

    payload = {
        "url": webhook_url,
        "update_types": ["message_created", "bot_started"],
        "secret": required_env("MAX_WEBHOOK_SECRET"),
    }

    try:
        async with httpx.AsyncClient(
            timeout=20,
            verify=max_ssl_context(),
        ) as http:
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
            detail={
                "status": response.status_code,
                "response": data,
            },
        )

    return {"ok": True, "response": data}
