# data/ingestion.py
import pydicom
from config.settings import MIMIC_CSV_PATH, MIMIC_IMAGES_PATH
import logging
import os
import pandas as pd
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from config.settings import DATA_DIR
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

class OpenIDataIngestion:
    """Data ingestion for OpenI chest X-ray dataset"""
    
    def __init__(self, data_dir=None):
        self.data_dir = data_dir or os.path.join(DATA_DIR, "raw")
        self.images_dir = os.path.join(self.data_dir, "images")
        self.reports_dir = os.path.join(self.data_dir, "reports")
        self.metadata_path = os.path.join(self.data_dir, "metadata.csv")
        print(f"Images directory: {self.images_dir}")
        print(f"Reports directory: {self.reports_dir}")
    
    def load_metadata(self, limit=None):
        """Load metadata CSV"""
        df = pd.read_csv(self.metadata_path)
        if limit:
            df = df.head(limit)
        return df
    
    def load_image(self, image_path):
        """Load image from file"""
        try:
            # For PNG files
            if image_path.lower().endswith('.png'):
                image = Image.open(image_path)
                image_array = np.array(image)
                return image_array
            # For DICOM files (if present)
            elif image_path.lower().endswith('.dcm'):
                import pydicom
                dicom = pydicom.dcmread(image_path)
                return dicom
            else:
                return None
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def parse_report(self, report_id):
        """Parse XML report file"""
        try:
            report_path = os.path.join(self.reports_dir, f"{report_id}.xml")
            tree = ET.parse(report_path)
            root = tree.getroot()
            
            # Extract relevant sections (this will vary based on XML structure)
            findings = root.find(".//findings").text if root.find(".//findings") is not None else ""
            impression = root.find(".//impression").text if root.find(".//impression") is not None else ""
            
            return {
                "findings": findings,
                "impression": impression,
                "full_text": f"{findings} {impression}".strip()
            }
        except Exception as e:
            print(f"Error parsing report {report_id}: {e}")
            return {"findings": "", "impression": "", "full_text": ""}
    
    def extract_batch(self, batch_size=100, offset=0):
        """Extract a batch of data for processing"""
        metadata = self.load_metadata()
        batch = metadata.iloc[offset:offset+batch_size]
        
        results = []
        for _, row in batch.iterrows():
            try:
                # Load image
                image_path = row["path"]
                print(f"Loading image from {image_path}")
                if not os.path.isabs(image_path):
                    image_path = os.path.join(self.data_dir, image_path)
                
                image_data = self.load_image(image_path)
                
                # Load report
                report_data = self.parse_report(row["report_id"])
                
                # Create result item
                if image_data is not None:
                    print(f"Processing image {image_path} and report {row['report_id']}")
                    results.append({
                        "image_id": row["image_id"],
                        "report_id": row["report_id"],
                        "path": image_path,
                        #"image": image_data,
                        "report": report_data["full_text"],
                        "findings": report_data["findings"],
                        "impression": report_data["impression"],
                        "finding_labels": row.get("finding", "").split(","),
                        "view_position": row.get("view_position", ""),
                        "patient_gender": row.get("patient_gender", ""),
                        "patient_age": row.get("patient_age", "")
                    })
            except Exception as e:
                print(f"Error processing row: {e}")
        
        return results

class MIMICCXRDataIngestion:
    """
    Data ingestion from MIMIC-CXR dataset. 
    Handles loading and basic extraction of metadata and image paths.
    """
    def __init__(self):
        self.metadata_path = MIMIC_CSV_PATH
        self.images_path = MIMIC_IMAGES_PATH
        
    def load_metadata(self, limit=None):
        """Load MIMIC-CXR metadata CSV"""
        logger.info(f"Loading metadata from {self.metadata_path}")
        try:
            df = pd.read_csv(self.metadata_path)
            if limit:
                df = df.head(limit)
            logger.info(f"Loaded {len(df)} metadata records")
            return df
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            raise
    
    def get_image_paths(self, df):
        """Map metadata to actual image file paths"""
        paths = []
        for _, row in df.iterrows():
            # Construct path based on MIMIC-CXR directory structure
            # Format: p10/p10000032/s55102753/02c5ee51-c644847a-35c3cf0e-33c6fb27-338bfd45.dcm
            subject_id = f"p{str(row['subject_id'])[:2]}/p{row['subject_id']}"
            study_id = f"s{row['study_id']}"
            dicom_id = row['dicom_id']
            path = os.path.join(self.images_path, subject_id, study_id, dicom_id)
            paths.append(path)
        return paths
    
    def load_dicom_image(self, path):
        """Load a DICOM image file"""
        try:
            dicom = pydicom.dcmread(path)
            return dicom
        except Exception as e:
            logger.error(f"Error loading DICOM {path}: {e}")
            return None
    
    def extract_batch(self, batch_size=100, offset=0):
        """Extract a batch of data for processing"""
        metadata = self.load_metadata()
        if offset >= len(metadata):
            logger.warning("Offset exceeds metadata length")
            return []
        batch = metadata.iloc[offset:offset+batch_size]
        image_paths = self.get_image_paths(batch)
        
        results = []
        '''
        for i, (_, row) in enumerate(batch.iterrows()):
            try:    
                dicom = self.load_dicom_image(image_paths[i])
                if dicom is not None:
                    results.append({
                        "subject_id": row["subject_id"],
                        "study_id": row["study_id"],
                        "dicom_id": row["dicom_id"],
                        "path": image_paths[i],
                        "dicom": dicom,
                        "report": row.get("report", ""),  # If report is available in metadata
                    })
            except Exception as e:
                logger.error(f"Error processing row {i}: {e}")
        '''
        for _, row in batch.iterrows():
            try:
                # Load image
                image_path = row["path"]
                if not os.path.isabs(image_path):
                    image_path = os.path.join(self.images_dir, image_path)
                
                image_data = self.load_image(image_path)
                
                # Create result item
                if image_data is not None:
                    results.append({
                        "image_id": row["image_id"],
                        "report_id": row["report_id"],
                        "path": image_path,
                        #"image": image_data,  # This should be "png" if that's what validation expects
                        "report": row["finding"],
                        "view_position": row["view_position"],
                        "patient_gender": row["patient_gender"],
                        "patient_age": row["patient_age"]
                    })
            except Exception as e:
                print(f"Error processing row: {e}")
        
        return results