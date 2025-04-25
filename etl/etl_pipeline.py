# etl/etl_pipeline.py
import logging
import os
import time
import random
import shutil
from config.settings import DATA_DIR
from data.ingestion import MIMICCXRDataIngestion, OpenIDataIngestion
from data.validation import DataValidator
from etl.kafka_producer import XrayKafkaProducer
from etl.kafka_consumer import XrayKafkaConsumer
from etl.spark_processor import SparkXrayProcessor

logger = logging.getLogger(__name__)

class XrayETLPipeline:
    """End-to-end ETL pipeline for X-ray data"""
    
    def __init__(self):
        #self.ingestion = MIMICCXRDataIngestion()
        self.ingestion = OpenIDataIngestion()
        self.validator = DataValidator()
        self.producer = XrayKafkaProducer()
        self.consumer = XrayKafkaConsumer()
        self.processor = SparkXrayProcessor()
        
        self.processed_dir = os.path.join(DATA_DIR, "processed")
        self.train_dir = os.path.join(self.processed_dir, "train")
        self.val_dir = os.path.join(self.processed_dir, "val")
        
        # Ensure directories exist
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)

    def _split_data(self, files, val_split=0.2):
        """Split processed data into train and validation sets"""
        logger.info(f"Splitting data with val_split={val_split}")
        
        # Shuffle files for random split
        random.shuffle(files)
        
        # Calculate split sizes
        val_size = int(len(files) * val_split)
        train_size = len(files) - val_size
        
        # Split files
        train_files = files[:train_size]
        val_files = files[train_size:]
        
        logger.info(f"Split sizes: train={len(train_files)}, val={len(val_files)}")
        
        # Move files to respective directories
        for file_path in train_files:
            if os.path.exists(file_path):
                filename = os.path.basename(file_path)
                dest_path = os.path.join(self.train_dir, filename)
                shutil.copy(file_path, dest_path)
                logger.debug(f"Copied {filename} to training set")
        
        for file_path in val_files:
            if os.path.exists(file_path):
                filename = os.path.basename(file_path)
                dest_path = os.path.join(self.val_dir, filename)
                shutil.copy(file_path, dest_path)
                logger.debug(f"Copied {filename} to validation set")
        
        logger.info(f"Moved {len(train_files)} files to training set")
        logger.info(f"Moved {len(val_files)} files to validation set")
    
    def run_ingestion_to_kafka(self, batch_size=100, num_batches=10):
        """Run data ingestion and push to Kafka"""
        total_records = 0
        valid_records = 0
        
        for i in range(num_batches):
            logger.info(f"Processing batch {i+1}/{num_batches}")
            
            # Extract batch from source
            batch_data = self.ingestion.extract_batch(
                batch_size=batch_size, 
                offset=i * batch_size
            )
            total_records += len(batch_data)
            
            # Validate data
            valid_data, invalid_data = self.validator.validate_batch(batch_data)
            valid_records += len(valid_data)
            
            # Push valid data to Kafka
            self.producer.send_batch(valid_data)
            
            logger.info(f"Batch {i+1} complete: {len(valid_data)}/{len(batch_data)} valid records")
        
        logger.info(f"Ingestion complete: {valid_records}/{total_records} valid records sent to Kafka")
        return valid_records
    
    def run_spark_processing(self, batch_size=1000, output_dir=None):
        """Consume from Kafka and process with Spark"""
        try:
            # Consume batch from Kafka
            records = self.consumer.consume_batch(max_records=batch_size)
            
            if not records:
                logger.warning("No records consumed from Kafka")
                return 0
            
            # Process with Spark
            processed_df = self.processor.process_batch(records)
            
            # Save processed data
            output_path = os.path.join(self.processed_dir, f"batch_{int(time.time())}")
            self.processor.save_processed_data(processed_df, output_path)
            
            count = processed_df.count()
            logger.info(f"Processed {count} records with Spark")
            return count
        
        except Exception as e:
            logger.error(f"Error in Spark processing: {e}")
            raise
        finally:
            # Clean up
            self.processor.stop()
    
    def run_pipeline(self, batch_size=100, num_batches=10, val_split=0.2):
        """Run the complete ETL pipeline"""
        try:
            # Step 1: Ingest data and push to Kafka
            records_ingested = self.run_ingestion_to_kafka(
                batch_size=batch_size, 
                num_batches=num_batches
            )
            
            # Step 2: Process data with Spark
            records_processed = self.run_spark_processing(
                batch_size=records_ingested
            )

            # Step 3: Split data into train and validation sets
            #files = [os.path.join(self.train_dir, f) for f in os.listdir(self.train_dir)]
            #self._split_data(files, val_split=val_split)
            #logger.info(f"Data split into train and validation sets")
            # Step 4: Clean up processed data
            #shutil.rmtree(self.processed_dir)
            #logger.info(f"Cleaned up processed data directory")
            
            logger.info(f"ETL pipeline complete: {records_processed} records processed")
            return records_processed
        
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise
        finally:
            # Clean up resources
            self.producer.close()
            self.consumer.close()