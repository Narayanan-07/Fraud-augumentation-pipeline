#!/bin/bash
# ─────────────────────────────────────────
# Start Kafka (Zookeeper + Broker + Topic)
# Run this FIRST before producer or stream processor
# Keep this terminal open
# ─────────────────────────────────────────

KAFKA_HOME="$HOME/kafka"
TOPIC="fraud-stream"

if [ ! -d "$KAFKA_HOME" ]; then
    echo "Kafka not found at $KAFKA_HOME. Run setup/install.sh first."
    exit 1
fi

echo "=== Starting Zookeeper ==="
$KAFKA_HOME/bin/zookeeper-server-start.sh \
    $KAFKA_HOME/config/zookeeper.properties &
ZOOKEEPER_PID=$!
echo "Zookeeper PID: $ZOOKEEPER_PID"
sleep 5

echo "=== Starting Kafka Broker ==="
$KAFKA_HOME/bin/kafka-server-start.sh \
    $KAFKA_HOME/config/server.properties &
BROKER_PID=$!
echo "Broker PID: $BROKER_PID"
sleep 6

echo "=== Creating Topic: $TOPIC ==="
$KAFKA_HOME/bin/kafka-topics.sh \
    --create \
    --if-not-exists \
    --topic $TOPIC \
    --bootstrap-server localhost:9092 \
    --partitions 1 \
    --replication-factor 1

echo ""
echo "=== Kafka Ready ==="
echo "Topic: $TOPIC"
echo "Broker: localhost:9092"
echo ""
echo "To stop: Ctrl+C then run setup/kafka_stop.sh"

# Keep script running — Ctrl+C to stop
wait $BROKER_PID
