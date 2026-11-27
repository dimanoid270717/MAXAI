import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import main


class TestGenerationSafety(unittest.IsolatedAsyncioTestCase):
    async def test_paid_generation_disabled_before_debit_or_provider_call(self):
        row = {"id": 17}
        with patch.object(main, "generation_enabled", return_value=False), \
             patch.object(main, "debit_generation") as debit, \
             patch.object(main, "Gen202Client") as provider:
            ok, message = await main.launch_generation(123, row)
        self.assertFalse(ok)
        self.assertIn("отключена", message)
        debit.assert_not_called()
        provider.assert_not_called()

    async def test_success_saves_first_valid_url_and_formats_newlines(self):
        row = {"id": 21, "provider_task_id": "task-21"}
        conn = MagicMock()
        conn.__enter__.return_value = conn
        conn.__exit__.return_value = False
        with patch.object(main, "Gen202Client") as provider, \
             patch.object(main, "db_connect", return_value=conn):
            provider.return_value.get_task = AsyncMock(return_value={
                "state": "success",
                "result_urls": ["not-a-url", "https://cdn.example/result.png",
                                "https://cdn.example/second.png"],
            })
            message = await main.refresh_generation(123, row)
        self.assertEqual(
            message,
            "Готово! Ссылка на результат:\n"
            "https://cdn.example/result.png\nhttps://cdn.example/second.png",
        )
        conn.execute.assert_called_once()
        self.assertIn("status='completed'", conn.execute.call_args.args[0])

    async def test_success_without_valid_url_does_not_mark_completed(self):
        row = {"id": 22, "provider_task_id": "task-22"}
        with patch.object(main, "Gen202Client") as provider, \
             patch.object(main, "db_connect") as db:
            provider.return_value.get_task = AsyncMock(return_value={
                "state": "success", "result_urls": ["javascript:bad"],
            })
            message = await main.refresh_generation(123, row)
        self.assertIn("не вернул ссылку", message)
        db.assert_not_called()

    async def test_failure_refunds_once_through_refund_handler(self):
        row = {"id": 23, "provider_task_id": "task-23"}
        with patch.object(main, "Gen202Client") as provider, \
             patch.object(main, "refund_generation", return_value=True) as refund:
            provider.return_value.get_task = AsyncMock(return_value={
                "state": "failed", "fail_message": "provider error",
            })
            message = await main.refresh_generation(123, row)
        self.assertIn("Кредиты возвращены", message)
        refund.assert_called_once_with(123, 23, "Ошибка GEN202: provider error")

    async def test_processing_state_does_not_change_credits_or_database(self):
        row = {"id": 24, "provider_task_id": "task-24"}
        with patch.object(main, "Gen202Client") as provider, \
             patch.object(main, "db_connect") as db, \
             patch.object(main, "refund_generation") as refund:
            provider.return_value.get_task = AsyncMock(return_value={"state": "processing"})
            message = await main.refresh_generation(123, row)
        self.assertIn("Генерация выполняется", message)
        db.assert_not_called()
        refund.assert_not_called()


    async def test_notification_marked_only_after_max_accepts_message(self):
        pending = [{
            "id": 31, "max_user_id": 123, "status": "completed",
            "result_url": "https://cdn.example/31.png", "last_error": None,
        }]
        conn = MagicMock()
        conn.__enter__.return_value = conn
        conn.__exit__.return_value = False
        conn.execute.return_value.fetchall.return_value = pending
        with patch.object(main, "db_connect", return_value=conn), \
             patch.object(main, "send_max", new=AsyncMock()) as send:
            await main.deliver_generation_notifications()
        send.assert_awaited_once()
        self.assertEqual(conn.execute.call_count, 2)
        self.assertIn("notification_sent_at=NOW()", conn.execute.call_args.args[0])

    async def test_notification_remains_pending_when_max_delivery_fails(self):
        pending = [{
            "id": 32, "max_user_id": 123, "status": "completed",
            "result_url": "https://cdn.example/32.png", "last_error": None,
        }]
        conn = MagicMock()
        conn.__enter__.return_value = conn
        conn.__exit__.return_value = False
        conn.execute.return_value.fetchall.return_value = pending
        with patch.object(main, "db_connect", return_value=conn), \
             patch.object(main, "send_max", new=AsyncMock(side_effect=RuntimeError("MAX unavailable"))):
            with self.assertRaisesRegex(RuntimeError, "MAX unavailable"):
                await main.deliver_generation_notifications()
        self.assertEqual(conn.execute.call_count, 1)


if __name__ == "__main__":
    unittest.main()
