from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Patient(Base):
    __tablename__ = 'patients'
    
    id = Column(String, primary_key=True)
    gender = Column(String)
    age = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    studies = relationship("Study", back_populates="patient")

class Study(Base):
    __tablename__ = 'studies'
    
    id = Column(String, primary_key=True)
    patient_id = Column(String, ForeignKey('patients.id'))
    study_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    patient = relationship("Patient", back_populates="studies")
    images = relationship("Image", back_populates="study")
    reports = relationship("Report", back_populates="study")

class Image(Base):
    __tablename__ = 'images'
    
    id = Column(String, primary_key=True)
    study_id = Column(String, ForeignKey('studies.id'))
    view_position = Column(String)
    file_path = Column(String)
    processed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    study = relationship("Study", back_populates="images")

class Report(Base):
    __tablename__ = 'reports'
    
    id = Column(String, primary_key=True)
    study_id = Column(String, ForeignKey('studies.id'))
    content = Column(Text)
    findings = Column(JSON)
    impression = Column(Text)
    generated_at = Column(DateTime, default=datetime.utcnow)
    confidence = Column(Float)
    model_version = Column(String)
    
    study = relationship("Study", back_populates="reports")

class ModelVersion(Base):
    __tablename__ = 'model_versions'
    
    id = Column(Integer, primary_key=True)
    version = Column(String, unique=True)
    mlflow_run_id = Column(String)
    metrics = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    deployed = Column(Boolean, default=False)