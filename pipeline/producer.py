"""
Kafka Producer — CSV Stream Replay
Person 1 owns this file.
Reads creditcard.csv and publishes rows to Kafka topic at controlled rate.
"""

import csv
import json
import time
import yaml
import logging
from pathlib import Path
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PRODUCER] %(message)s"
)
log = logging.getLogger(__name__)


def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


def create_producer(broker: str) -> KafkaProducer:
    try:
        producer = KafkaProducer(
            bootstrap_servers=[broker],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: str(k).encode("utf-8")
        )
        log.info(f"Connected to Kafka broker: {broker}")
        return producer
    except NoBrokersAvailable:
        log.error("Kafka broker not available. Run setup/kafka_start.sh first.")
        raise


def stream_csv(producer: KafkaProducer, config: dict):
    topic = config["kafka"]["topic"]
    rate = config["kafka"]["records_per_second"]
    delay = 1.0 / rate

    # Use sample data if raw data not available
    raw_path = Path(config["paths"]["raw_data"])
    sample_path = Path(config["paths"]["sample_data"])

    if raw_path.exists():
        data_path = raw_path
        log.info(f"Streaming from full dataset: {raw_path}")
    elif sample_path.exists():
        data_path = sample_path
        log.warning(f"Raw data not found. Streaming from sample: {sample_path}")
    else:
        log.error("No data file found. Place creditcard.csv in data/raw/")
        raise FileNotFoundError("No data source available.")

    sent = 0
    loop = 0

    log.info(f"Starting stream to topic '{topic}' at {rate} records/sec")
    log.info("Press Ctrl+C to stop.")

    try:
        while True:  # Loop through dataset continuously
            loop += 1
            with open(data_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert all values to float where possible
                    record = {}
                    for k, v in row.items():
                        try:
                            record[k] = float(v)
                        except ValueError:
                            record[k] = v

                    record["_loop"] = loop
                    record["_sent_at"] = time.time()

                    producer.send(
                        topic=topic,
                        key=str(sent),
                        value=record
                    )

                    sent += 1
                    if sent % 100 == 0:
                        log.info(f"Sent {sent} records | Loop {loop} | "
                                 f"Class: {int(record.get('Class', 0))}")

                    time.sleep(delay)

            log.info(f"Completed loop {loop}. Restarting stream...")

    except KeyboardInterrupt:
        log.info(f"Stream stopped. Total records sent: {sent}")
    finally:
        producer.flush()
        producer.close()
        log.info("Producer closed.")


if __name__ == "__main__":
    cfg = load_config()
    prod = create_producer(cfg["kafka"]["broker"])
    stream_csv(prod, cfg)
