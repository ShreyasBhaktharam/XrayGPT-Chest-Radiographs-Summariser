from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
import os
from .models import Base

class DatabaseManager:
    def __init__(self):
        self.engine = create_engine(os.getenv('POSTGRES_CONNECTION'))
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def init_db(self):
        """Initialize database tables"""
        Base.metadata.create_all(bind=self.engine)
    
    @contextmanager
    def get_session(self):
        """Provide a transactional scope for operations"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def save_image_record(self, image_data):
        """Save image metadata to database"""
        from .models import Image
        
        with self.get_session() as session:
            image = Image(
                id=image_data['id'],
                study_id=image_data['study_id'],
                view_position=image_data['view_position'],
                file_path=image_data['file_path']
            )
            session.add(image)
    
    def save_report(self, report_data):
        """Save generated report to database"""
        from .models import Report
        
        with self.get_session() as session:
            report = Report(
                id=report_data['id'],
                study_id=report_data['study_id'],
                content=report_data['content'],
                findings=report_data['findings'],
                impression=report_data['impression'],
                confidence=report_data['confidence'],
                model_version=report_data['model_version']
            )
            session.add(report)