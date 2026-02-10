import json
import os
import signal
import statistics
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Tuple

from kafka import KafkaConsumer, TopicPartition
from kafka.structs import OffsetAndMetadata

BROKER = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "redpanda:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "sensor-events")
GROUP_ID = os.getenv("KAFKA_CONSUMER_GROUP", "feature-agg-v1")
AUTO_OFFSET_RESET = os.getenv("KAFKA_AUTO_OFFSET_RESET", "latest")

WINDOW_SHORT_SECONDS = float(os.getenv("WINDOW_SHORT_SECONDS", "5"))
WINDOW_LONG_SECONDS = float(os.getenv("WINDOW_LONG_SECONDS", "30"))
FLUSH_INTERVAL_SECONDS = float(os.getenv("FLUSH_INTERVAL_SECONDS", "5"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/data/features_offline"))

REPORT_INTERVAL_SECONDS = float(os.getenv("REPORT_INTERVAL_SECONDS", "10"))
MAX_RUNTIME_SECONDS = float(os.getenv("MAX_RUNTIME_SECONDS", "0"))

METRICS = {
    "temperature": "temp",
    "vibration": "vib",
    "pressure": "pressure",
    "rpm": "rpm",
    "load": "load",
}

running = True


def handle_signal(_sig, _frame):
    global running
    running = False


def now_epoch() -> float:
    return time.time()


def epoch_to_iso(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()


def parse_event_ts(raw_ts) -> float:
    if raw_ts is None:
        return now_epoch()
    if isinstance(raw_ts, (int, float)):
        return float(raw_ts)
    if isinstance(raw_ts, str):
        try:
            parsed = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
        except ValueError:
            return now_epoch()
    return now_epoch()


def numeric(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def window_suffix(seconds: float) -> str:
    if float(seconds).is_integer():
        return f"{int(seconds)}s"
    return f"{str(seconds).replace('.', '_')}s"


def prune_old(buffer: Deque[Tuple[float, dict]], latest_event_ts: float) -> None:
    cutoff = latest_event_ts - WINDOW_LONG_SECONDS
    while buffer and buffer[0][0] < cutoff:
        buffer.popleft()


def filter_window(buffer: Deque[Tuple[float, dict]], latest_event_ts: float, window_seconds: float) -> List[Tuple[float, dict]]:
    cutoff = latest_event_ts - window_seconds
    return [item for item in buffer if item[0] >= cutoff]


def slope(points_x: List[float], points_y: List[float]) -> float:
    n = len(points_x)
    if n < 2:
        return 0.0

    x0 = points_x[0]
    shifted_x = [x - x0 for x in points_x]

    sx = sum(shifted_x)
    sy = sum(points_y)
    sxx = sum(x * x for x in shifted_x)
    sxy = sum(x * y for x, y in zip(shifted_x, points_y))

    denom = (n * sxx) - (sx * sx)
    if denom == 0:
        return 0.0

    return ((n * sxy) - (sx * sy)) / denom


def metric_stats(window_items: List[Tuple[float, dict]], metric: str) -> Dict[str, float]:
    if not window_items:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "slope": 0.0,
        }

    xs = [ts for ts, _ in window_items]
    ys = [numeric(event.get(metric)) for _, event in window_items]

    mean_val = sum(ys) / len(ys)
    std_val = statistics.pstdev(ys) if len(ys) > 1 else 0.0

    return {
        "mean": round(mean_val, 6),
        "std": round(std_val, 6),
        "min": round(min(ys), 6),
        "max": round(max(ys), 6),
        "slope": round(slope(xs, ys), 6),
    }


def build_feature_row(
    machine_id: str,
    event_ts: float,
    ingest_ts: float,
    latest_event: dict,
    short_items: List[Tuple[float, dict]],
    long_items: List[Tuple[float, dict]],
) -> dict:
    short_suffix = window_suffix(WINDOW_SHORT_SECONDS)
    long_suffix = window_suffix(WINDOW_LONG_SECONDS)

    row = {
        "machine_id": machine_id,
        "event_ts": epoch_to_iso(event_ts),
        "ingest_ts": epoch_to_iso(ingest_ts),
        "window_short_seconds": WINDOW_SHORT_SECONDS,
        "window_long_seconds": WINDOW_LONG_SECONDS,
        "temperature": numeric(latest_event.get("temperature")),
        "vibration": numeric(latest_event.get("vibration")),
        "pressure": numeric(latest_event.get("pressure")),
        "rpm": numeric(latest_event.get("rpm")),
        "load": numeric(latest_event.get("load")),
    }

    row[f"event_count_{short_suffix}"] = len(short_items)
    row[f"event_count_{long_suffix}"] = len(long_items)

    for metric, alias in METRICS.items():
        short_stats = metric_stats(short_items, metric)
        long_stats = metric_stats(long_items, metric)

        for stat_name, stat_value in short_stats.items():
            row[f"{alias}_{stat_name}_{short_suffix}"] = stat_value
        for stat_name, stat_value in long_stats.items():
            row[f"{alias}_{stat_name}_{long_suffix}"] = stat_value

    return row


def write_rows_atomic(rows: List[dict]) -> Path:
    flush_ts = datetime.now(timezone.utc)
    partition_dir = OUTPUT_DIR / flush_ts.strftime("dt=%Y-%m-%d") / flush_ts.strftime("hour=%H")
    partition_dir.mkdir(parents=True, exist_ok=True)

    filename = f"part-{int(flush_ts.timestamp() * 1000)}"
    temp_path = partition_dir / f"{filename}.jsonl.tmp"
    final_path = partition_dir / f"{filename}.jsonl"

    with temp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")))
            handle.write("\n")

    temp_path.replace(final_path)
    return final_path


def create_consumer() -> KafkaConsumer:
    while True:
        try:
            consumer = KafkaConsumer(
                TOPIC,
                bootstrap_servers=BROKER,
                group_id=GROUP_ID,
                enable_auto_commit=False,
                auto_offset_reset=AUTO_OFFSET_RESET,
                value_deserializer=lambda payload: json.loads(payload.decode("utf-8")),
                key_deserializer=lambda payload: payload.decode("utf-8") if payload else None,
                consumer_timeout_ms=1000,
            )
            print(
                f"connected to kafka broker={BROKER} topic={TOPIC} group={GROUP_ID} auto_offset_reset={AUTO_OFFSET_RESET}"
            )
            return consumer
        except Exception as exc:
            print(f"waiting for kafka broker: {exc}")
            time.sleep(2)


def flush_and_commit(consumer: KafkaConsumer, pending_rows: List[dict], pending_offsets: Dict[TopicPartition, int]) -> bool:
    if not pending_rows:
        return True

    try:
        file_path = write_rows_atomic(pending_rows)
        commit_map = {
            # kafka-python in current images expects (offset, metadata, leader_epoch)
            topic_partition: OffsetAndMetadata(offset, "", -1)
            for topic_partition, offset in pending_offsets.items()
        }
        consumer.commit(offsets=commit_map)
        print(f"flushed rows={len(pending_rows)} file={file_path}")
        pending_rows.clear()
        pending_offsets.clear()
        return True
    except Exception as exc:
        print(f"flush failed, will retry rows={len(pending_rows)} error={exc}")
        return False


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    consumer = create_consumer()
    start_monotonic = time.monotonic()

    machine_buffers: Dict[str, Deque[Tuple[float, dict]]] = defaultdict(deque)
    pending_rows: List[dict] = []
    pending_offsets: Dict[TopicPartition, int] = {}

    last_flush = time.monotonic()
    last_report = time.monotonic()
    processed_total = 0

    while running:
        polled = consumer.poll(timeout_ms=1000, max_records=500)

        for topic_partition, records in polled.items():
            for record in records:
                event = record.value
                if not isinstance(event, dict):
                    continue

                machine_id = event.get("machine_id") or record.key
                if not machine_id:
                    continue

                event_ts = parse_event_ts(event.get("ts"))
                ingest_ts = now_epoch()

                buffer = machine_buffers[machine_id]
                buffer.append((event_ts, event))
                prune_old(buffer, event_ts)

                short_items = filter_window(buffer, event_ts, WINDOW_SHORT_SECONDS)
                long_items = filter_window(buffer, event_ts, WINDOW_LONG_SECONDS)

                row = build_feature_row(
                    machine_id=machine_id,
                    event_ts=event_ts,
                    ingest_ts=ingest_ts,
                    latest_event=event,
                    short_items=short_items,
                    long_items=long_items,
                )

                pending_rows.append(row)
                pending_offsets[topic_partition] = record.offset + 1
                processed_total += 1

        now_monotonic = time.monotonic()

        if now_monotonic - last_flush >= FLUSH_INTERVAL_SECONDS:
            if flush_and_commit(consumer, pending_rows, pending_offsets):
                last_flush = now_monotonic

        if now_monotonic - last_report >= REPORT_INTERVAL_SECONDS:
            print(
                f"status processed_total={processed_total} pending_rows={len(pending_rows)} tracked_machines={len(machine_buffers)}"
            )
            last_report = now_monotonic

        if MAX_RUNTIME_SECONDS > 0 and (now_monotonic - start_monotonic) >= MAX_RUNTIME_SECONDS:
            print(f"max runtime reached ({MAX_RUNTIME_SECONDS}s); stopping aggregator")
            break

    flush_and_commit(consumer, pending_rows, pending_offsets)
    consumer.close()
    print("aggregator stopped")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    main()
