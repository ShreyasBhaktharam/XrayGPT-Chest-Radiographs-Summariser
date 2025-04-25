# etl/kafka_consumer.py
import json
import logging
import pickle
import base64
from kafka import KafkaConsumer
from config.settings import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC_RAW

logger = logging.getLogger(__name__)

class XrayKafkaConsumer:
    """Consumes X-ray data from Kafka topic"""
    
    def __init__(self, group_id='xraygpt-consumer'):
        self.consumer = KafkaConsumer(
            KAFKA_TOPIC_RAW,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            group_id=group_id,
            auto_offset_reset='earliest',
            value_deserializer=self._deserialize
        )
    
    def _deserialize(self, data):
        """Deserialize data from Kafka - decode base64 and unpickle binary data"""
        try:
            # Parse JSON string
            json_data = json.loads(data.decode('utf-8'))
            
            # Handle binary data (DICOM)
            if "dicom" in json_data:
                # Decode base64 string
                base64_dicom = json_data["dicom"]
                # Then unpickle to get DICOM object
                pickled_dicom = base64.b64decode(base64_dicom)
                json_data["dicom"] = pickle.loads(pickled_dicom)
            
            return json_data
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            raise
    
    def consume_batch(self, max_records=100, timeout_ms=10000):
        """Consume a batch of records from Kafka"""
        batch = []
        
        # Poll for messages
        messages = self.consumer.poll(timeout_ms=timeout_ms, max_records=max_records)
        
        for topic_partition, records in messages.items():
            for record in records:
                batch.append(record.value)
        
        logger.info(f"Consumed {len(batch)} records from Kafka")
        return batch
    
    def subscribe(self):
        """Subscribe to topic and yield records as they arrive"""
        try:
            for record in self.consumer:
                yield record.value
        except Exception as e:
            logger.error(f"Error in consumer subscription: {e}")
            raise
    
    def close(self):
        """Close Kafka consumer connection"""
        self.consumer.close()