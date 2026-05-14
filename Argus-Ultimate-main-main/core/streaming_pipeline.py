"""
Kafka Streaming Pipeline for Argus Ultimate
===========================================
Streams real-time alternative data into Argus:
- Twitter sentiment
- On-chain metrics (e.g., whale movements)
- Satellite imagery (e.g., oil tanker counts)
- News sentiment

Dependencies:
- kafka-python
- pandas
- numpy
"""

import json
import logging
import threading
from typing import Dict, List, Optional, Callable, Any
from collections import deque
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class KafkaStreamer:
    """
    Kafka-based streaming pipeline for alternative data.
    """
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        group_id: str = "argus-group",
        auto_offset_reset: str = "latest",
    ):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.auto_offset_reset = auto_offset_reset

        # Kafka consumers for different data sources
        self.consumers: Dict[str, KafkaConsumer] = {}
        self.producer: Optional[KafkaProducer] = None

        # Data buffers (thread-safe)
        self.data_buffers: Dict[str, deque] = {}
        self.buffer_locks: Dict[str, threading.Lock] = {}
        self.buffer_sizes: Dict[str, int] = {
            "twitter_sentiment": 1000,
            "on_chain": 1000,
            "satellite": 100,
            "news_sentiment": 1000,
        }

        # Confidence scores for each data source
        self.confidence_scores: Dict[str, float] = {
            "twitter_sentiment": 0.7,
            "on_chain": 0.9,
            "satellite": 0.8,
            "news_sentiment": 0.75,
        }

        # Callbacks for new data
        self.callbacks: Dict[str, List[Callable[[Dict], None]]] = {}

        # Initialize
        self._init_kafka()

    def _init_kafka(self):
        """Initialize Kafka consumers and producer."""
        try:
            # Twitter sentiment consumer
            self.consumers["twitter_sentiment"] = KafkaConsumer(
                "twitter_sentiment",
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset=self.auto_offset_reset,
                value_deserializer=lambda x: json.loads(x.decode("utf-8")),
            )

            # On-chain metrics consumer
            self.consumers["on_chain"] = KafkaConsumer(
                "on_chain_metrics",
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset=self.auto_offset_reset,
                value_deserializer=lambda x: json.loads(x.decode("utf-8")),
            )

            # Satellite imagery consumer
            self.consumers["satellite"] = KafkaConsumer(
                "satellite_data",
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset=self.auto_offset_reset,
                value_deserializer=lambda x: json.loads(x.decode("utf-8")),
            )

            # News sentiment consumer
            self.consumers["news_sentiment"] = KafkaConsumer(
                "news_sentiment",
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset=self.auto_offset_reset,
                value_deserializer=lambda x: json.loads(x.decode("utf-8")),
            )

            # Producer for internal communication
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda x: json.dumps(x).encode("utf-8"),
            )

            # Initialize buffers
            for topic in self.buffer_sizes:
                self.data_buffers[topic] = deque(maxlen=self.buffer_sizes[topic])
                self.buffer_locks[topic] = threading.Lock()

            logger.info("KafkaStreamer initialized with topics: " + ", ".join(self.consumers.keys()))

        except KafkaError as e:
            logger.error(f"Failed to initialize Kafka: {e}")
            raise

    def start(self):
        """Start consuming data from all topics."""
        for topic, consumer in self.consumers.items():
            thread = threading.Thread(
                target=self._consume_topic,
                args=(topic, consumer),
                daemon=True,
            )
            thread.start()
            logger.info(f"Started Kafka consumer for topic: {topic}")

    def _consume_topic(self, topic: str, consumer: KafkaConsumer):
        """Consume messages from a Kafka topic."""
        try:
            for message in consumer:
                data = message.value
                with self.buffer_locks[topic]:
                    self.data_buffers[topic].append(data)

                # Notify callbacks
                if topic in self.callbacks:
                    for callback in self.callbacks[topic]:
                        try:
                            callback(data)
                        except Exception as e:
                            logger.error(f"Callback error for {topic}: {e}")

        except KafkaError as e:
            logger.error(f"Kafka consumer error for {topic}: {e}")

    def get_data(self, topic: str, limit: Optional[int] = None) -> List[Dict]:
        """Get data from a topic's buffer."""
        with self.buffer_locks.get(topic, threading.Lock()):
            if limit is None:
                return list(self.data_buffers.get(topic, []))
            else:
                return list(self.data_buffers.get(topic, []))[-limit:]

    def get_fused_data(self) -> Dict[str, Any]:
        """
        Get fused data from all topics, weighted by confidence.
        Returns:
            Dict with fused data and metadata
        """
        fused_data = {}
        metadata = {}

        for topic, confidence in self.confidence_scores.items():
            data = self.get_data(topic, limit=1)
            if data:
                fused_data[topic] = data[0]
                metadata[topic] = {
                    "confidence": confidence,
                    "timestamp": data[0].get("timestamp"),
                }

        return {
            "data": fused_data,
            "metadata": metadata,
        }

    def register_callback(self, topic: str, callback: Callable[[Dict], None]):
        """Register a callback for new data on a topic."""
        if topic not in self.callbacks:
            self.callbacks[topic] = []
        self.callbacks[topic].append(callback)

    def send_data(self, topic: str, data: Dict):
        """Send data to a Kafka topic."""
        if self.producer:
            try:
                self.producer.send(topic, data)
                self.producer.flush()
            except KafkaError as e:
                logger.error(f"Failed to send data to {topic}: {e}")

    def close(self):
        """Close all Kafka connections."""
        for consumer in self.consumers.values():
            consumer.close()
        if self.producer:
            self.producer.close()
        logger.info("KafkaStreamer closed")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    streamer = KafkaStreamer(bootstrap_servers="localhost:9092")

    def twitter_callback(data: Dict):
        print(f"New Twitter data: {data}")

    streamer.register_callback("twitter_sentiment", twitter_callback)
    streamer.start()

    try:
        while True:
            fused = streamer.get_fused_data()
            if fused["data"]:
                print(f"Fused data: {fused}")
    except KeyboardInterrupt:
        streamer.close()
