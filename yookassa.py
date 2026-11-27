import os
import uuid
from decimal import Decimal, InvalidOperation

import httpx
import psycopg
from fastapi import HTTPException

YOOKASSA_API = "https://api.yookassa.ru/v3"


def _credentials() -> tuple[str, str]:
    shop_id = os.getenv("YOOKASSA_SHOP_ID", "").strip()
    secret = os.getenv("YOOKASSA_SECRET_KEY", "").strip()
    if not shop_id or not secret:
        raise HTTPException(status_code=503, detail="ЮKassa не настроена")
    return shop_id, secret


async def create_yookassa_payment(
    *,
    order_code: str,
    amount_rub: int,
    credits: int,
) -> tuple[str, str]:
    """Создаёт платёж и возвращает (payment_id, confirmation_url)."""
    shop_id, secret = _credentials()
    payload = {
        "amount": {
            "value": f"{Decimal(amount_rub):.2f}",
            "currency": "RUB",
        },
        "capture": True,
        "confirmation": {
            "type": "redirect",
            "return_url": "https://maxai-tls6.onrender.com/payment/return",
        },
        "description": f"NEUROPIX: {credits} кредитов, заказ {order_code}",
        "metadata": {
            "order_code": order_code,
            "credits": str(credits),
        },
    }
    try:
        async with httpx.AsyncClient(timeout=25) as client:
            response = await client.post(
                f"{YOOKASSA_API}/payments",
                auth=(shop_id, secret),
                headers={"Idempotence-Key": str(uuid.uuid4())},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        # Не передаём пользователю секреты и полный ответ провайдера.
        raise HTTPException(
            status_code=502,
            detail=f"ЮKassa отклонила создание платежа (HTTP {exc.response.status_code})",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail="Не удалось связаться с ЮKassa",
        ) from exc

    payment_id = data.get("id")
    confirmation_url = (data.get("confirmation") or {}).get("confirmation_url")
    if not payment_id or not confirmation_url:
        raise HTTPException(
            status_code=502,
            detail="ЮKassa вернула неполные данные платежа",
        )
    return str(payment_id), str(confirmation_url)


async def fetch_yookassa_payment(payment_id: str) -> dict:
    """Получает актуальный платёж напрямую из API ЮKassa."""
    shop_id, secret = _credentials()
    try:
        async with httpx.AsyncClient(timeout=25) as client:
            response = await client.get(
                f"{YOOKASSA_API}/payments/{payment_id}",
                auth=(shop_id, secret),
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail="Не удалось проверить платёж в ЮKassa",
        ) from exc


def save_payment_id(database_url: str, order_code: str, payment_id: str) -> None:
    with psycopg.connect(database_url) as conn:
        row = conn.execute(
            """
            UPDATE purchase_orders
            SET provider_payment_id = %s,
                payment_provider = 'yookassa',
                updated_at = NOW()
            WHERE order_code = %s
              AND status = 'awaiting_payment'
            RETURNING id
            """,
            (payment_id, order_code),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=409, detail="Заявка не найдена или уже обработана")


def credit_paid_order(database_url: str, payment: dict) -> bool:
    """
    Зачисляет кредиты только после проверки платежа через API ЮKassa.
    Возвращает True, если это первое успешное зачисление.
    """
    if payment.get("status") != "succeeded" or payment.get("paid") is not True:
        return False

    payment_id = str(payment.get("id") or "")
    metadata = payment.get("metadata") or {}
    order_code = str(metadata.get("order_code") or "")
    if not payment_id or not order_code:
        return False

    try:
        paid_amount = Decimal(str((payment.get("amount") or {}).get("value")))
    except (InvalidOperation, TypeError):
        return False
    if (payment.get("amount") or {}).get("currency") != "RUB":
        return False

    with psycopg.connect(database_url) as conn:
        order = conn.execute(
            """
            SELECT id, max_user_id, package_credits, amount_rub,
                   status, provider_payment_id
            FROM purchase_orders
            WHERE order_code = %s
            FOR UPDATE
            """,
            (order_code,),
        ).fetchone()

        if not order:
            return False
        if order["provider_payment_id"] != payment_id:
            return False
        if paid_amount != Decimal(order["amount_rub"]).quantize(Decimal("0.01")):
            return False
        if int(metadata.get("credits", "-1")) != int(order["package_credits"]):
            return False
        if order["status"] == "paid":
            return False
        if order["status"] != "awaiting_payment":
            return False

        conn.execute(
            """
            UPDATE users
            SET balance = balance + %s,
                last_seen_at = NOW()
            WHERE max_user_id = %s
            """,
            (int(order["package_credits"]), int(order["max_user_id"])),
        )
        conn.execute(
            """
            INSERT INTO credit_transactions
                (max_user_id, amount, reason, idempotency_key)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (idempotency_key) WHERE idempotency_key IS NOT NULL
            DO NOTHING
            """,
            (
                int(order["max_user_id"]),
                int(order["package_credits"]),
                f"Оплата ЮKassa, заказ {order_code}",
                f"yookassa:{payment_id}",
            ),
        )
        conn.execute(
            """
            UPDATE purchase_orders
            SET status = 'paid', updated_at = NOW()
            WHERE id = %s AND status = 'awaiting_payment'
            """,
            (int(order["id"]),),
        )
        return True
