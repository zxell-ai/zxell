from sqlalchemy import Column, Integer, String, Text
from database import Base

class TrainingData(Base):
    __tablename__ = "training_data"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    category = Column(String, nullable=True)
    content = Column(Text, nullable=False)
    assigned_to = Column(String, nullable=True)
    status = Column(String, default="pending")  # pending, processing, done

