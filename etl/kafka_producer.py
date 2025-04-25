# etl/kafka_producer.py
import json
import logging
from kafka import KafkaProducer
from config.settings import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC_RAW
import pickle
import base64

logger = logging.getLogger(__name__)

class XrayKafkaProducer:
    """Produces X-ray data to Kafka topic"""
    
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=self._serialize,
            max_request_size=10485760,  # 10 MB
        )
        self.topic = KAFKA_TOPIC_RAW
        
    def _serialize(self, data):
        """Serialize data for Kafka - pickle binary data and encode as base64"""
        try:
            # Handle binary data (DICOM)
            if "dicom" in data:
                # First pickle the DICOM object
                pickled_dicom = pickle.dumps(data["dicom"])
                # Then encode as base64 string
                data["dicom"] = base64.b64encode(pickled_dicom).decode('ascii')

            # Handle image data (PNG)
            if "image" in data:
                print(f"Serializing png image: {data['image']}")
                # First pickle the image object
                pickled_image = pickle.dumps(data["image"])
                # Then encode as base64 string
                data["image"] = base64.b64encode(pickled_image).decode('ascii')
            
            # Convert to JSON string
            return json.dumps(data).encode('utf-8')
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            raise
    
    def send_record(self, data):
        """Send a single record to Kafka"""
        try:
            # Using image_id + report_id as key for partitioning
            key = f"{data['image_id']}_{data['report_id']}".encode('utf-8')
            print(f"Key: {key}")
            print(f"Data: {data}")
            future = self.producer.send(self.topic, key=key, value=data)
            # Wait for message to be sent
            future.get(timeout=10)
            logger.debug(f"Sent record to Kafka: {data['image_id']}")
            return True
        except Exception as e:
            logger.error(f"Error sending to Kafka: {e}")
            return False
    
    def send_batch(self, batch_data):
        """Send batch of records to Kafka"""
        success_count = 0
        for item in batch_data:
            if self.send_record(item):
                success_count += 1
        
        logger.info(f"Published {success_count}/{len(batch_data)} records to Kafka")
        return success_count
    
    def close(self):
        """Close Kafka producer connection"""
        self.producer.close()