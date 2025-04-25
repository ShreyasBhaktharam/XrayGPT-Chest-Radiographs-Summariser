from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Metrics for ETL pipeline
etl_processed_records = Counter('xraygpt_etl_processed_records_total', 'Total records processed by ETL')
etl_processing_time = Histogram('xraygpt_etl_processing_seconds', 'Time spent processing records')
etl_errors = Counter('xraygpt_etl_errors_total', 'Total ETL errors', ['error_type'])

# Metrics for model training
training_loss = Gauge('xraygpt_training_loss', 'Current training loss')
training_accuracy = Gauge('xraygpt_training_accuracy', 'Current training accuracy')
training_epoch = Gauge('xraygpt_training_epoch', 'Current training epoch')

# Metrics for inference
inference_requests = Counter('xraygpt_inference_requests_total', 'Total inference requests')
inference_latency = Histogram('xraygpt_inference_latency_seconds', 'Inference latency')
model_confidence = Histogram('xraygpt_model_confidence', 'Model prediction confidence')

class MetricsCollector:
    def __init__(self, port=8000):
        self.port = port
        
    def start(self):
        """Start metrics server"""
        start_http_server(self.port)
    
    @staticmethod
    def record_etl_processing(func):
        """Decorator to measure ETL processing time"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                etl_processed_records.inc()
                return result
            except Exception as e:
                etl_errors.labels(error_type=type(e).__name__).inc()
                raise
            finally:
                etl_processing_time.observe(time.time() - start_time)
        return wrapper
    
    @staticmethod
    def record_inference(func):
        """Decorator to measure inference metrics"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            inference_requests.inc()
            inference_latency.observe(time.time() - start_time)
            if 'confidence' in result:
                model_confidence.observe(result['confidence'])
            return result
        return wrapper