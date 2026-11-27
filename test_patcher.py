import unittest, tempfile, os, runpy, contextlib, io
from pathlib import Path

PATCHER = Path(__file__).resolve().parent / "apply_neuropix_2.py"
MAIN = """from fastapi import FastAPI, HTTPException
from gen202 import Gen202Client, Gen202Error
app = FastAPI()
@app.get("/tasks/{task_id}")
async def task_status(task_id: str):
    try:
        return await Gen202Client().get_task(task_id)
    except Gen202Error as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
@app.get("/payment/return")
async def payment_return():
    return {}
"""
YK = """import uuid
async def create_yookassa_payment(order_code, payment_method):
    headers={"Idempotence-Key": str(uuid.uuid4())}
async def fetch_yookassa_refund(refund_id: str) -> dict:
    return {}
async def fetch_yookassa_refund(refund_id: str) -> dict:
    return {}
"""

class TestPatcher(unittest.TestCase):
    def test_safe_patch(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p/"apply_neuropix_2.py").write_text(PATCHER.read_text(encoding="utf-8"), encoding="utf-8")
            (p/"main.py").write_text(MAIN, encoding="utf-8")
            (p/"yookassa.py").write_text(YK, encoding="utf-8")
            old = os.getcwd()
            try:
                os.chdir(p)
                with contextlib.redirect_stdout(io.StringIO()):
                    runpy.run_path(str(p/"apply_neuropix_2.py"), run_name="__main__")
            finally:
                os.chdir(old)
            self.assertNotIn('/tasks/{task_id}', (p/"main.py").read_text())
            y = (p/"yookassa.py").read_text()
            self.assertIn("uuid.uuid5", y)
            self.assertEqual(y.count("async def fetch_yookassa_refund"), 1)
            self.assertIn("ENABLE_PAID_GENERATION=false", (p/".env.example").read_text())
            self.assertTrue(list(p.glob(".neuropix-backup-*")))

if __name__ == "__main__":
    unittest.main()
