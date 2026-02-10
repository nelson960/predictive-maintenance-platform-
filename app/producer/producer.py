import json
import os
import random
import time
from datetime import datetime, timezone

from kafka import KafkaProducer

BROKER = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "redpanda:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "sensor-events")
NUM_MACHINES = int(os.getenv("NUM_MACHINES", "50"))
TICK_SECONDS = float(os.getenv("TICK_SECONDS", "1"))
MAX_RUNTIME_SECONDS = float(os.getenv("MAX_RUNTIME_SECONDS", "0"))


def runtime_exceeded(start_monotonic: float) -> bool:
    return MAX_RUNTIME_SECONDS > 0 and (time.monotonic() - start_monotonic) >= MAX_RUNTIME_SECONDS


while True:
    try:
        producer = KafkaProducer(
            bootstrap_servers=BROKER,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            retries=5,
        )
        break
    except Exception:
        time.sleep(2)

machines = [f"M-{i:03d}" for i in range(1, NUM_MACHINES + 1)]
start_monotonic = time.monotonic()

while True:
    if runtime_exceeded(start_monotonic):
        print(f"max runtime reached ({MAX_RUNTIME_SECONDS}s); stopping producer")
        break

    ts = datetime.now(timezone.utc).isoformat()
    for m in machines:
        event = {
            "machine_id": m,
            "ts": ts,
            "temperature": round(random.normalvariate(72, 2), 2),
            "vibration": round(random.normalvariate(0.25, 0.05), 3),
            "pressure": round(random.normalvariate(35, 1.5), 2),
            "rpm": int(random.normalvariate(1800, 80)),
            "load": round(random.uniform(0.4, 0.9), 3),
        }
        producer.send(TOPIC, key=m.encode("utf-8"), value=event)
    producer.flush()
    print(f"sent batch at {ts}")

    if runtime_exceeded(start_monotonic):
        print(f"max runtime reached ({MAX_RUNTIME_SECONDS}s); stopping producer")
        break

    time.sleep(TICK_SECONDS)

producer.close()
print("producer stopped")
