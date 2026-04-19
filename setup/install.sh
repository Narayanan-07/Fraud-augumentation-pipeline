#!/bin/bash
# ─────────────────────────────────────────
# One-script Ubuntu environment setup
# Run once: bash setup/install.sh
# ─────────────────────────────────────────

set -e
echo "=== Fraud Augmentation Pipeline — Environment Setup ==="

# System packages
echo "[1/6] Installing system packages..."
sudo apt update -q
sudo apt install -y python3-pip python3-venv openjdk-11-jdk wget curl unzip

# Verify Java
echo "[2/6] Checking Java..."
java -version
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
echo "JAVA_HOME=$JAVA_HOME"

# Python virtual environment
echo "[3/6] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q

# Python dependencies
echo "[4/6] Installing Python packages (this takes 3-5 mins)..."
pip install -r requirements.txt -q
echo "Python packages installed."

# Kafka download
echo "[5/6] Setting up Kafka..."
KAFKA_VERSION="3.7.0"
SCALA_VERSION="2.13"
KAFKA_DIR="$HOME/kafka"

if [ ! -d "$KAFKA_DIR" ]; then
    wget -q "https://downloads.apache.org/kafka/${KAFKA_VERSION}/kafka_${SCALA_VERSION}-${KAFKA_VERSION}.tgz"
    tar -xzf "kafka_${SCALA_VERSION}-${KAFKA_VERSION}.tgz"
    mv "kafka_${SCALA_VERSION}-${KAFKA_VERSION}" "$KAFKA_DIR"
    rm "kafka_${SCALA_VERSION}-${KAFKA_VERSION}.tgz"
    echo "Kafka installed at $KAFKA_DIR"
else
    echo "Kafka already installed at $KAFKA_DIR"
fi

# Add Kafka to PATH
if ! grep -q "KAFKA_HOME" ~/.bashrc; then
    echo '' >> ~/.bashrc
    echo '# Kafka' >> ~/.bashrc
    echo "export KAFKA_HOME=$KAFKA_DIR" >> ~/.bashrc
    echo 'export PATH=$PATH:$KAFKA_HOME/bin' >> ~/.bashrc
    echo "Added Kafka to PATH in ~/.bashrc"
fi

# Create required directories
echo "[6/6] Creating runtime directories..."
mkdir -p tmp/spark_checkpoint \
         data/raw data/processed data/augmented data/models \
         evaluation/results logs

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  source venv/bin/activate"
echo "  source ~/.bashrc"
echo "  bash setup/kafka_start.sh      (in a separate terminal)"
echo "  python pipeline/producer.py    (in a separate terminal)"
echo "  python pipeline/stream_processor.py"
