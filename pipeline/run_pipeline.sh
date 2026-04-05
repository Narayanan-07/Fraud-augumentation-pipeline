#!/bin/bash
# ─────────────────────────────────────────
# Run full pipeline (after Kafka is running)
# Usage: bash pipeline/run_pipeline.sh
# ─────────────────────────────────────────

source venv/bin/activate 2>/dev/null || true

echo "=== Starting Fraud Augmentation Pipeline ==="
echo "Kafka must already be running (bash setup/kafka_start.sh)"
echo ""

# Start producer in background
echo "[1/2] Starting Kafka producer..."
python pipeline/producer.py &
PRODUCER_PID=$!
echo "Producer PID: $PRODUCER_PID"
sleep 3

# Start stream processor
echo "[2/2] Starting stream processor..."
echo "Press Ctrl+C to stop both."
python pipeline/stream_processor.py

# Cleanup on exit
kill $PRODUCER_PID 2>/dev/null
echo "Pipeline stopped."
