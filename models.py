from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# SQLAlchemyのベースクラスを作成
Base = declarative_base()


class Diagnosis(Base):
    """診断結果の永続化テーブル"""

    __tablename__ = "diagnoses"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    image_path = Column(String, nullable=False)
    image_hash = Column(String, index=True)
    mime_type = Column(String)
    file_size = Column(Integer)
    diagnosis_text = Column(Text, nullable=False)
    model_name = Column(String)
    model_description = Column(String)
    processing_time = Column(Float)

    chat_messages = relationship(
        "ChatMessage",
        back_populates="diagnosis",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class ChatMessage(Base):
    """診断に紐づくチャット履歴"""

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    diagnosis_id = Column(Integer, ForeignKey("diagnoses.id", ondelete="CASCADE"))
    role = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    diagnosis = relationship("Diagnosis", back_populates="chat_messages")


class Feedback(Base):
    """任意のフィードバックを保存するテーブル"""

    __tablename__ = "feedbacks"

    id = Column(Integer, primary_key=True, index=True)
    diagnosis_id = Column(Integer, ForeignKey("diagnoses.id", ondelete="SET NULL"))
    user_feedback = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
