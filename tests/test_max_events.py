import unittest

from main import max_event_key


class TestMaxEventKey(unittest.TestCase):
    def test_uses_unique_message_mid(self):
        update = {
            "update_type": "message_created",
            "message": {"body": {"mid": "mid_123"}},
        }
        self.assertEqual(max_event_key(update), "max:message_created:mid_123")

    def test_accepts_mid_on_message_object(self):
        update = {
            "update_type": "message_created",
            "message": {"mid": "mid_456"},
        }
        self.assertEqual(max_event_key(update), "max:message_created:mid_456")

    def test_ignores_other_event_types(self):
        self.assertIsNone(max_event_key({"update_type": "bot_started"}))

    def test_missing_mid_falls_back_to_no_dedupe_key(self):
        self.assertIsNone(max_event_key({
            "update_type": "message_created",
            "message": {"body": {"text": "hello"}},
        }))


if __name__ == "__main__":
    unittest.main()
