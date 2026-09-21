from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default="student")  # admin, faculty, student
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class DocumentRecord(Base):
    __tablename__ = "documents"

    source_id = Column(String(255), primary_key=True, index=True)
    file_name = Column(String(255), nullable=False)
    uploaded_by = Column(String(255), nullable=False)
    visibility = Column(String(50), nullable=False, default="public")  # public, private
    file_path = Column(String(1024), nullable=True)
    chunks_count = Column(Integer, default=0, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String(255), nullable=False, index=True)
    user_role = Column(String(50), nullable=True)
    question = Column(Text, nullable=False)
    source_id = Column(String(255), nullable=True)
    num_contexts = Column(Integer, default=0, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
