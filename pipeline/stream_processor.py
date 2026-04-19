"""
Stream Processor — PySpark Structured Streaming from Kafka.
Person 1 owns this file.
This is the integration point where Person 2's augment() is called.
"""

import json
import time
import yaml
import logging
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, schema_of_json
from pyspark.sql.types import (
    StructType, StructField, DoubleType, IntegerType, StringType
)

from pipeline.preprocessor import preprocess, load_schema
from pipeline.storage_writer import write_processed, write_augmented

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [STREAM] %(message)s"
)
log = logging.getLogger(__name__)


def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


def build_kafka_schema():
    """Spark schema matching creditcard.csv columns as strings from Kafka JSON."""
    fields = (
        [StructField(f"V{i}", DoubleType(), True) for i in range(1, 29)] +
        [
            StructField("Amount", DoubleType(), True),
            StructField("Class", DoubleType(), True),
            StructField("Time", DoubleType(), True),
            StructField("_loop", IntegerType(), True),
            StructField("_sent_at", DoubleType(), True),
        ]
    )
    return StructType(fields)


def get_augmentor():
    """
    Load Person 2's augmentation function.
    Falls back to passthrough if augmentation module not yet built.
    """
    try:
        from augmentation.augmentor import augment
        log.info("Augmentation module loaded: augmentation/augmentor.py")
        return augment
    except ImportError:
        log.warning("augmentation/augmentor.py not found. Running WITHOUT augmentation.")
        log.warning("Person 2: implement augment(df, batch_id) -> df in augmentation/augmentor.py")

        def passthrough(df: pd.DataFrame, batch_id: int) -> pd.DataFrame:
            df["is_synthetic"] = False
            df["batch_id"] = batch_id
            return df

        return passthrough


def make_batch_processor(config: dict, schema: dict, augment_fn):
    """Returns the foreachBatch function for Spark."""

    def process_batch(spark_df, batch_id: int):
        start = time.time()

        if spark_df.isEmpty():
            log.warning(f"Batch {batch_id}: empty — skipping.")
            return

        # Convert Spark DF to Pandas for preprocessing and augmentation
        pandas_df = spark_df.toPandas()
        log.info(f"Batch {batch_id}: received {len(pandas_df)} raw records")

        # Step 1 — Preprocess
        clean_df = preprocess(pandas_df, config, schema)
        if clean_df.empty:
            log.warning(f"Batch {batch_id}: no valid rows after preprocessing.")
            return

        # Step 2 — Write original processed batch
        write_processed(clean_df, batch_id, config)

        # Step 3 — Augment (Person 2's function)
        try:
            augmented_df = augment_fn(clean_df.copy(), batch_id)
        except Exception as e:
            log.error(f"Augmentation failed on batch {batch_id}: {e}")
            augmented_df = clean_df.copy()
            augmented_df["is_synthetic"] = False
            augmented_df["batch_id"] = batch_id

        # Step 4 — Write augmented batch
        write_augmented(augmented_df, batch_id, config)

        elapsed = time.time() - start
        synthetic_count = augmented_df.get("is_synthetic", pd.Series()).sum()
        log.info(f"Batch {batch_id} done | {len(clean_df)} real + "
                 f"{synthetic_count} synthetic | {elapsed:.2f}s")

    return process_batch


def run():
    config = load_config()
    schema = load_schema()

    spark = (
        SparkSession.builder
        .appName(config["pipeline"]["spark_app_name"])
        .master(config["pipeline"]["spark_master"])
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel(config["pipeline"]["log_level"])
    log.info("Spark session started.")

    kafka_schema = build_kafka_schema()
    augment_fn = get_augmentor()

    # Read stream from Kafka
    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", config["kafka"]["broker"])
        .option("subscribe", config["kafka"]["topic"])
        .option("startingOffsets", config["kafka"]["auto_offset_reset"])
        .load()
    )

    # Parse JSON value from Kafka
    parsed_stream = (
        raw_stream
        .select(
            from_json(col("value").cast("string"), kafka_schema).alias("data")
        )
        .select("data.*")
    )

    batch_processor = make_batch_processor(config, schema, augment_fn)

    query = (
    parsed_stream.writeStream
    .foreachBatch(batch_processor)
    .option("checkpointLocation", config["pipeline"]["checkpoint_dir"])
    .option("failOnDataLoss", "false")
    .trigger(processingTime=f"{config['kafka']['batch_interval_seconds']} seconds")
    .start()
    )

    log.info("Stream processor running. Press Ctrl+C to stop.")
    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        log.info("Stopping stream processor...")
        query.stop()
        spark.stop()


if __name__ == "__main__":
    run()
