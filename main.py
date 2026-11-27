import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel, Field, HttpUrl
import httpx
from gen202 import Gen202Client, Gen202Error

load_dotenv()
app = FastAPI(title="NEUROPIX AI bridge", version="0.2.0")
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

def gen_client():
    try:
        return Gen202Client()
    except Gen202Error as exc:
        raise HTTPException(status_code=503, detail=str(exc))

def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise HTTPException(status_code=503, detail=f"{name} is not configured")
    return value

async def send_max(user_id: int, text: str):
    async with httpx.AsyncClient(timeout=20) as http:
        r = await http.post(
            f"{MAX_API}/messages",
            params={"user_id": user_id},
            headers={"Authorization": required_env("MAX_BOT_TOKEN")},
            json={"text": text[:4000]},
        )
    if r.is_error:
        raise HTTPException(status_code=502, detail=f"MAX API returned {r.status_code}")

@app.get("/health")
async def health():
    return {"ok": True, "service": "NEUROPIX AI bridge"}

@app.post("/generate/image")
async def generate_image(body: ImageRequest):
    try:
        return await gen_client().create_image(
            prompt=body.prompt, aspect_ratio=body.aspect_ratio,
            resolution=body.resolution, output_format=body.output_format,
            image_input=[str(x) for x in body.image_input] if body.image_input else None,
        )
    except Gen202Error as exc:
        raise HTTPException(status_code=502, detail=str(exc))

@app.post("/generate/video")
async def generate_video(body: VideoRequest):
    try:
        return await gen_client().create_video(
            prompt=body.prompt, duration=body.duration, aspect_ratio=body.aspect_ratio,
            resolution=body.resolution, generate_audio=body.generate_audio,
            first_frame_url=str(body.first_frame_url) if body.first_frame_url else None,
            last_frame_url=str(body.last_frame_url) if body.last_frame_url else None,
            reference_image_urls=[str(x) for x in body.reference_image_urls] if body.reference_image_urls else None,
            reference_video_urls=[str(x) for x in body.reference_video_urls] if body.reference_video_urls else None,
            reference_audio_urls=[str(x) for x in body.reference_audio_urls] if body.reference_audio_urls else None,
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
async def max_webhook(request: Request, x_max_bot_api_secret: str | None = Header(default=None)):
    if x_max_bot_api_secret != required_env("MAX_WEBHOOK_SECRET"):
        raise HTTPException(status_code=401, detail="Invalid webhook secret")
    update = await request.json()
    kind = update.get("update_type")
    if kind == "bot_started":
        user_id = (update.get("user") or {}).get("user_id")
        if user_id:
            await send_max(int(user_id), "Добро пожаловать в NEUROPIX!\nСоздаю ИИ-фото и видео. Напиши /help.")
        return {"ok": True}
    if kind != "message_created":
        return {"ok": True}
    message = update.get("message") or {}
    user_id = (message.get("sender") or {}).get("user_id")
    body = message.get("body") or {}
    text = (body.get("text") or message.get("text") or "").strip()
    if not user_id or not text:
        return {"ok": True}
    cmd = text.lower()
    if cmd in ("/start", "старт", "начать"):
        answer = "NEUROPIX — ИИ-фото и видео.\n\nНапиши /help для инструкции."
    elif cmd in ("/help", "помощь"):
        answer = "Отправь запрос в формате:\nФото: портрет в студии...\nВидео: кинематографичная сцена...\n\nОплата, баланс и полноценная генерация из чата подключаются отдельно."
    elif cmd.startswith(("фото:", "image:")):
        answer = "Запрос принят. Следующим этапом подключим генерацию и отправку готового изображения прямо сюда."
    elif cmd.startswith(("видео:", "video:")):
        answer = "Запрос принят. Следующим этапом подключим генерацию и отправку готового видео прямо сюда."
    else:
        answer = "Я NEUROPIX — бот для ИИ-фото и видео. Напиши /help."
    await send_max(int(user_id), answer)
    return {"ok": True}

@app.post("/max/register-webhook")
async def register_webhook(x_admin_secret: str | None = Header(default=None)):
    if x_admin_secret != required_env("MAX_ADMIN_SECRET"):
        raise HTTPException(status_code=401, detail="Invalid admin secret")
    url = required_env("MAX_WEBHOOK_URL")
    if not url.startswith("https://"):
        raise HTTPException(status_code=500, detail="MAX_WEBHOOK_URL must use HTTPS")
    payload = {
        "url": url,
        "update_types": ["message_created", "bot_started"],
        "secret": required_env("MAX_WEBHOOK_SECRET"),
    }
    async with httpx.AsyncClient(timeout=20) as http:
        r = await http.post(
            f"{MAX_API}/subscriptions",
            headers={"Authorization": required_env("MAX_BOT_TOKEN"), "Content-Type": "application/json"},
            json=payload,
        )
    try:
        data = r.json()
    except ValueError:
        data = {"raw": r.text[:400]}
    if r.is_error or data.get("success") is False:
        raise HTTPException(status_code=502, detail={"status": r.status_code, "response": data})
    return {"ok": True, "response": data}
