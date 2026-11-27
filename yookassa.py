import os
import uuid
from decimal import Decimal, InvalidOperation

import httpx
import psycopg
from psycopg.rows import dict_row
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
    payment_method: str = "bank_card",
) -> tuple[str, str]:
    """Создаёт платёж и возвращает (payment_id, confirmation_url)."""
    shop_id, secret = _credentials()
    if payment_method not in ("bank_card", "sbp"):
        raise HTTPException(status_code=400, detail="Неподдерживаемый способ оплаты")
    payload = {
        "amount": {
            "value": f"{Decimal(amount_rub):.2f}",
            "currency": "RUB",
        },
        "capture": True,
        "payment_method_data": {"type": payment_method},
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


async def fetch_yookassa_refund(refund_id: str) -> dict:
    """Получает актуальный возврат напрямую из API ЮKassa."""
    shop_id, secret = _credentials()
    try:
        async with httpx.AsyncClient(timeout=25) as client:
            response = await client.get(
                f"{YOOKASSA_API}/refunds/{refund_id}",
                auth=(shop_id, secret),
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail="Не удалось проверить возврат в ЮKassa",
        ) from exc


def mark_payment_cancelled(database_url: str, payment: dict) -> bool:
    """Помечает заказ отменённым только по подтверждённому статусу API."""
    if payment.get("status") != "canceled":
        return False
    payment_id = str(payment.get("id") or "")
    if not payment_id:
        return False
    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        row = conn.execute(
            """UPDATE purchase_orders
               SET status='cancelled', updated_at=NOW()
               WHERE provider_payment_id=%s AND status='awaiting_payment'
               RETURNING id""",
            (payment_id,),
        ).fetchone()
        return bool(row)


def process_yookassa_refund(database_url: str, refund: dict) -> str:
    """Обрабатывает полный возврат. Частичный/нехватка баланса требуют ручной проверки."""
    if refund.get("status") != "succeeded":
        return "ignored"
    refund_id = str(refund.get("id") or "")
    payment_id = str(refund.get("payment_id") or "")
    if not refund_id or not payment_id:
        return "ignored"
    try:
        refund_amount = Decimal(str((refund.get("amount") or {}).get("value")))
    except (InvalidOperation, TypeError):
        return "ignored"
    if (refund.get("amount") or {}).get("currency") != "RUB":
        return "ignored"

    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        order = conn.execute(
            """SELECT id, max_user_id, package_credits, amount_rub,
                      status, provider_payment_id
               FROM purchase_orders WHERE provider_payment_id=%s FOR UPDATE""",
            (payment_id,),
        ).fetchone()
        if not order or order["status"] != "paid":
            return "ignored"
        if order.get("provider_refund_id") == refund_id or order["status"] == "refunded":
            return "duplicate"
        prior = conn.execute(
            "SELECT id FROM credit_transactions WHERE idempotency_key=%s",
            (f"yookassa-refund:{refund_id}",),
        ).fetchone()
        if prior:
            return "duplicate"

        full_amount = Decimal(order["amount_rub"]).quantize(Decimal("0.01"))
        if refund_amount != full_amount:
            conn.execute(
                "UPDATE purchase_orders SET status='refund_review', provider_refund_id=%s, updated_at=NOW() WHERE id=%s",
                (refund_id, int(order["id"])),
            )
            return "review"

        user = conn.execute(
            "SELECT balance FROM users WHERE max_user_id=%s FOR UPDATE",
            (int(order["max_user_id"]),),
        ).fetchone()
        credits = int(order["package_credits"])
        if not user or int(user["balance"]) < credits:
            conn.execute(
                "UPDATE purchase_orders SET status='refund_review', provider_refund_id=%s, updated_at=NOW() WHERE id=%s",
                (refund_id, int(order["id"])),
            )
            return "review"

        conn.execute(
            "UPDATE users SET balance=balance-%s, last_seen_at=NOW() WHERE max_user_id=%s",
            (credits, int(order["max_user_id"])),
        )
        inserted = conn.execute(
            """INSERT INTO credit_transactions
                   (max_user_id, amount, reason, idempotency_key)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (idempotency_key) WHERE idempotency_key IS NOT NULL
               DO NOTHING RETURNING id""",
            (int(order["max_user_id"]), -credits,
             f"Возврат ЮKassa, заказ по платежу {payment_id}",
             f"yookassa-refund:{refund_id}"),
        ).fetchone()
        if not inserted:
            # На случай гонки: откатить изменение баланса текущей транзакции.
            raise RuntimeError("Повторный возврат: операция уже зарегистрирована")
        conn.execute(
            "UPDATE purchase_orders SET status='refunded', provider_refund_id=%s, updated_at=NOW() WHERE id=%s",
            (refund_id, int(order["id"])),
        )
        return "refunded"


def save_payment_id(database_url: str, order_code: str, payment_id: str) -> None:
    with psycopg.connect(database_url, row_factory=dict_row) as conn:
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

    with psycopg.connect(database_url, row_factory=dict_row) as conn:
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

        inserted = conn.execute(
            """
            INSERT INTO credit_transactions
                (max_user_id, amount, reason, idempotency_key)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (idempotency_key) WHERE idempotency_key IS NOT NULL
            DO NOTHING
            RETURNING id
            """,
            (
                int(order["max_user_id"]),
                int(order["package_credits"]),
                f"Оплата ЮKassa, заказ {order_code}",
                f"yookassa:{payment_id}",
            ),
        ).fetchone()
        if not inserted:
            return False
        updated = conn.execute(
            """
            UPDATE users
            SET balance = balance + %s,
                last_seen_at = NOW()
            WHERE max_user_id = %s
            RETURNING max_user_id
            """,
            (int(order["package_credits"]), int(order["max_user_id"])),
        ).fetchone()
        if not updated:
            raise RuntimeError("Пользователь для начисления не найден")
        conn.execute(
            """
            UPDATE purchase_orders
            SET status = 'paid', updated_at = NOW()
            WHERE id = %s AND status = 'awaiting_payment'
            """,
            (int(order["id"]),),
        )
        return True
