#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime, timezone
import shutil, sys

ROOT = Path(__file__).resolve().parent
BACKUP = ROOT / (".neuropix-backup-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))

def fail(message):
    print("ОШИБКА:", message, file=sys.stderr)
    raise SystemExit(1)

def main():
    mp, yp, ep = ROOT/"main.py", ROOT/"yookassa.py", ROOT/".env.example"
    if not mp.is_file() or not yp.is_file():
        fail("Положи скрипт в корень репозитория рядом с main.py и yookassa.py.")
    m, y = mp.read_text(encoding="utf-8"), yp.read_text(encoding="utf-8")
    route_start = m.find('@app.get("/tasks/{task_id}")')
    route_end = m.find('@app.get("/payment/return")', route_start)
    key = 'headers={"Idempotence-Key": str(uuid.uuid4())}'
    matches = [i for i, line in enumerate(y.splitlines(keepends=True))
               if line.startswith("async def fetch_yookassa_refund(")]
    if len(matches) != 2:
        fail(f"Ожидалось ровно два определения fetch_yookassa_refund(), найдено {len(matches)}.")
    y_lines = y.splitlines(keepends=True)
    b = sum(len(line) for line in y_lines[:matches[1]])
    BACKUP.mkdir()
    shutil.copy2(mp, BACKUP/"main.py")
    shutil.copy2(yp, BACKUP/"yookassa.py")
    if ep.exists(): shutil.copy2(ep, BACKUP/".env.example")
    m = m[:route_start] + m[route_end:]
    y = y[:b].rstrip() + "\n"
    y = y.replace(key, 'headers={"Idempotence-Key": str(uuid.uuid5(uuid.NAMESPACE_URL, f"neuropix:yookassa:{order_code}:{payment_method}"))}', 1)
    mp.write_text(m, encoding="utf-8")
    yp.write_text(y, encoding="utf-8")
    ep.write_text("# NEUROPIX — пример переменных окружения; не коммить реальные секреты.\n"
        "DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/DBNAME\n"
        "MAX_BOT_TOKEN=\nMAX_WEBHOOK_SECRET=\n"
        "MAX_WEBHOOK_URL=https://YOUR-SERVICE.onrender.com/webhook/max\n"
        "MAX_ADMIN_SECRET=\nMAX_CA_FILE=\n"
        "YOOKASSA_SHOP_ID=\nYOOKASSA_SECRET_KEY=\n"
        "GEN202_API_KEY=\nGEN202_BASE_URL=https://api.gen202.com\n"
        "GEN202_IMAGE_MODEL=nano-banana/2-1\nGEN202_VIDEO_MODEL=seedance/2-5\n"
        "# Оставить выключенным до отдельного разрешения владельца.\n"
        "ENABLE_PAID_GENERATION=false\n", encoding="utf-8")
    print("Готово: закрыт публичный task endpoint; стабилизирован ключ платежа; обновлён .env.example.")
    print("Резервные копии:", BACKUP.name)
    print("Платная генерация не включалась. Деплой не выполнялся.")

if __name__ == "__main__":
    main()
