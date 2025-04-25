# etl/spark_processor.py
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, udf, lit
from pyspark.sql.types import StringType, StructType, StructField, IntegerType, BinaryType
import numpy as np
import pydicom
from io import BytesIO
import base64
import pickle
from config.settings import SPARK_MASTER, SPARK_APP_NAME

logger = logging.getLogger(__name__)

class SparkXrayProcessor:
    """Process X-ray images using Spark"""
    
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName(SPARK_APP_NAME) \
            .master(SPARK_MASTER) \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.memory", "2g") \
            .getOrCreate()
        
        logger.info(f"Initialized Spark session: {self.spark.version}")
    
    def create_dataframe_from_records(self, records):
        """Create Spark DataFrame from Kafka records"""
        # Define schema for the DataFrame
        schema = StructType([
            StructField("subject_id", StringType(), True),
            StructField("study_id", StringType(), True),
            StructField("dicom_id", StringType(), True),
            StructField("path", StringType(), True),
            # We'll store the dicom data as serialized binary
            StructField("dicom_binary", BinaryType(), True),
            StructField("report", StringType(), True),
        ])
        
        # Convert records to rows
        rows = []
        for record in records:
            # Serialize DICOM object to binary
            dicom_binary = pickle.dumps(record.get("dicom"))
            
            row = (
                str(record.get("subject_id")),
                str(record.get("study_id")),
                record.get("dicom_id"),
                record.get("path"),
                dicom_binary, 
                record.get("report", "")
            )
            rows.append(row)
        
        # Create DataFrame
        df = self.spark.createDataFrame(rows, schema=schema)
        return df
    
    # UDFs for image processing
    @staticmethod
    @udf(returnType=BinaryType())
    def extract_pixel_data(dicom_binary):
        """Extract pixel data from DICOM binary"""
        try:
            dicom = pickle.loads(dicom_binary)
            # Convert to float and normalize to 0-1
            pixel_array = dicom.pixel_array.astype(np.float32)
            if pixel_array.max() > 0:
                pixel_array = pixel_array / pixel_array.max()
            # Convert back to bytes for storage
            return pickle.dumps(pixel_array)
        except:
            return None
    
    @staticmethod
    @udf(returnType=StringType())
    def extract_dicom_metadata(dicom_binary):
        """Extract metadata from DICOM binary as JSON string"""
        try:
            dicom = pickle.loads(dicom_binary)
            metadata = {
                "PatientID": str(getattr(dicom, "PatientID", "")),
                "PatientSex": str(getattr(dicom, "PatientSex", "")),
                "PatientAge": str(getattr(dicom, "PatientAge", "")),
                "ViewPosition": str(getattr(dicom, "ViewPosition", "")),
                "Rows": int(getattr(dicom, "Rows", 0)),
                "Columns": int(getattr(dicom, "Columns", 0)),
            }
            return json.dumps(metadata)
        except:
            return "{}"
    
    def process_batch(self, records):
        """Process a batch of X-ray records"""
        # Create DataFrame from records
        df = self.create_dataframe_from_records(records)
        
        # Apply transformations
        processed_df = df \
            .withColumn("pixel_data", self.extract_pixel_data(col("dicom_binary"))) \
            .withColumn("metadata", self.extract_dicom_metadata(col("dicom_binary"))) \
            .drop("dicom_binary")  # Remove original binary data to save memory
        
        # Cache the processed DataFrame to improve performance
        processed_df.cache()
        
        return processed_df
    
    def save_processed_data(self, df, output_path):
        """Save processed data to parquet files"""
        try:
            df.write.mode("append").parquet(output_path)
            logger.info(f"Saved processed data to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            return False
    
    def stop(self):
        """Stop Spark session"""
        self.spark.stop()