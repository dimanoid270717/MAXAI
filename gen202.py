
import json
import os
import asyncio
import ssl
from typing import Any

import certifi
import httpx


class Gen202Error(RuntimeError):
    pass


class Gen202Client:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEN202_API_KEY", "").strip()
        self.base_url = os.getenv(
            "GEN202_BASE_URL",
            "https://api.gen202.com"
        ).rstrip("/")

        self.image_model = os.getenv(
            "GEN202_IMAGE_MODEL",
            "nano-banana/2-1"
        )

        self.video_model = os.getenv(
            "GEN202_VIDEO_MODEL",
            "seedance/2-5"
        )

        # Используем системный набор доверенных сертификатов certifi.
        # Проверка SSL остаётся включённой.
        self.ssl_context = ssl.create_default_context(
            cafile=certifi.where()
        )

        if not self.api_key or self.api_key == "PASTE_KEY_HERE":
            raise Gen202Error(
                "GEN202_API_KEY не задан в переменных окружения"
            )

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def create_task(
        self,
        model: str,
        input_data: dict[str, Any]
    ) -> dict[str, Any]:

        async with httpx.AsyncClient(
            timeout=45,
            verify=self.ssl_context
        ) as client:

            response = await client.post(
                f"{self.base_url}/api/v1/jobs/createTask",
                headers=self.headers,
                json={
                    "model": model,
                    "input": input_data
                },
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise Gen202Error(
                f"GEN202 вернул не JSON (HTTP {response.status_code})"
            ) from exc

        if response.is_error or payload.get("code") not in (200, None):
            raise Gen202Error(
                str(payload.get("msg") or payload)
            )

        task_id = (payload.get("data") or {}).get("taskId")

        if not task_id:
            raise Gen202Error(
                f"В ответе GEN202 нет taskId: {payload}"
            )

        return {
            "task_id": task_id,
            "model": model
        }

    async def create_image(
        self,
        prompt: str,
        aspect_ratio: str = "1:1",
        resolution: str = "1K",
        output_format: str = "jpg",
        image_input: list[str] | None = None
    ) -> dict[str, Any]:

        data: dict[str, Any] = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "output_format": output_format,
        }

        if image_input:
            data["image_input"] = image_input

        return await self.create_task(
            self.image_model,
            data
        )

    async def create_video(
        self,
        prompt: str,
        duration: int = 5,
        aspect_ratio: str = "9:16",
        resolution: str = "480p",
        generate_audio: bool = True,
        first_frame_url: str | None = None,
        last_frame_url: str | None = None,
        reference_image_urls: list[str] | None = None,
        reference_video_urls: list[str] | None = None,
        reference_audio_urls: list[str] | None = None
    ) -> dict[str, Any]:

        if not 4 <= duration <= 30:
            raise Gen202Error(
                "Seedance 2.5: длительность должна быть от 4 до 30 секунд"
            )

        data: dict[str, Any] = {
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "generate_audio": generate_audio,
        }

        optional_fields = (
            ("first_frame_url", first_frame_url),
            ("last_frame_url", last_frame_url),
            ("reference_image_urls", reference_image_urls),
            ("reference_video_urls", reference_video_urls),
            ("reference_audio_urls", reference_audio_urls),
        )

        for key, value in optional_fields:
            if value:
                data[key] = value

        return await self.create_task(
            self.video_model,
            data
        )

    async def get_task(
        self,
        task_id: str
    ) -> dict[str, Any]:

        async with httpx.AsyncClient(
            timeout=30,
            verify=self.ssl_context
        ) as client:

            response = await client.get(
                f"{self.base_url}/api/v1/jobs/recordInfo",
                headers=self.headers,
                params={"taskId": task_id},
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise Gen202Error(
                f"GEN202 вернул не JSON (HTTP {response.status_code})"
            ) from exc

        if response.is_error or payload.get("code") not in (200, None):
            raise Gen202Error(
                str(payload.get("msg") or payload)
            )

        data = payload.get("data") or {}

        result: dict[str, Any] = {
            "task_id": data.get("taskId", task_id),
            "state": data.get("state"),
            "fail_code": data.get("failCode"),
            "fail_message": data.get("failMsg"),
            "credits_consumed": data.get("creditsConsumed"),
        }

        raw_result = data.get("resultJson")

        if raw_result:
            try:
                parsed = (
                    json.loads(raw_result)
                    if isinstance(raw_result, str)
                    else raw_result
                )

                result["result_urls"] = parsed.get(
                    "resultUrls",
                    []
                )

            except (ValueError, AttributeError):
                result["result_urls"] = []

        return result

    async def wait_for_task(
        self,
        task_id: str,
        timeout_seconds: int = 3600
    ) -> dict[str, Any]:

        deadline = (
            asyncio.get_running_loop().time()
            + timeout_seconds
        )

        while asyncio.get_running_loop().time() < deadline:

            result = await self.get_task(task_id)

            if result["state"] in ("success", "fail"):
                return result

            await asyncio.sleep(3)

        raise Gen202Error(
            "Истекло время ожидания задачи"
        )
