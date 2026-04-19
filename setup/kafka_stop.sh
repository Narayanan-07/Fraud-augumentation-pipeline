#!/bin/bash
KAFKA_HOME="$HOME/kafka"
echo "Stopping Kafka..."
$KAFKA_HOME/bin/kafka-server-stop.sh
sleep 3
echo "Stopping Zookeeper..."
$KAFKA_HOME/bin/zookeeper-server-stop.sh
echo "Kafka stopped."
