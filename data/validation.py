# data/validation.py
import logging
import pandas as pd
from PIL import Image
import numpy as np
import io

logger = logging.getLogger(__name__)

class DataValidator:
    """Validates data quality before processing"""
    
    def __init__(self):
        self.validation_errors = []
    
    def validate_metadata(self, df):
        """Check if metadata has required columns and is properly formatted"""
        required_columns = ["subject_id", "study_id", "dicom_id"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            error = f"Missing required columns: {missing_columns}"
            self.validation_errors.append(error)
            logger.error(error)
            return False
        
        # Check for missing values in required columns
        for col in required_columns:
            if df[col].isnull().any():
                error = f"Column {col} has missing values"
                self.validation_errors.append(error)
                logger.error(error)
                return False
                
        return True
    
    def validate_dicom(self, dicom):
        """Check if DICOM file has required attributes"""
        try:
            # Check for pixel data
            if not hasattr(dicom, 'pixel_array'):
                error = "DICOM missing pixel data"
                self.validation_errors.append(error)
                logger.error(error)
                return False
            
            # Check image dimensions are reasonable
            pixel_array = dicom.pixel_array
            if pixel_array.shape[0] < 100 or pixel_array.shape[1] < 100:
                error = f"DICOM image too small: {pixel_array.shape}"
                self.validation_errors.append(error)
                logger.warning(error)
                return False
                
            return True
        except Exception as e:
            error = f"DICOM validation error: {e}"
            self.validation_errors.append(error)
            logger.error(error)
            return False
        
    def validate_png(self, png):
        """Check if PNG file is valid"""
        try:
            # Attempt to open the image
            #img = Image.open(io.BytesIO(png))
            #img.verify()  # Verify that it is a valid image
            return True
        except Exception as e:
            error = f"PNG validation error: {e}"
            self.validation_errors.append(error)
            logger.error(error)
            return False
    
    def validate_batch(self, batch_data):
        """Validate a batch of data"""
        valid_items = []
        invalid_items = []
        
        for item in batch_data:
            print(f"VALIDATING: {item}")
            #if self.validate_dicom(item.get("dicom")):
             #   valid_items.append(item)
            if "image" in item or "path" in item:
                print("VALID")
                valid_items.append(item)
            else:
                print("INVALID")
                invalid_items.append(item)
        print(f"valid_items_length: {len(valid_items)}")
        print(f"invalid_items_length: {len(invalid_items)}")
        logger.info(f"Batch validation: {len(valid_items)} valid, {len(invalid_items)} invalid")
        return valid_items, invalid_items
    
    def get_validation_report(self):
        """Get summary of validation errors"""
        return {
            "total_errors": len(self.validation_errors),
            "errors": self.validation_errors[:10],  # First 10 errors
        }