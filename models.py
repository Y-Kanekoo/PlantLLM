from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

# SQLAlchemyのベースクラスを作成
Base = declarative_base()


class Diagnosis(Base):
    """
    病害診断結果を保存するテーブル

    Attributes:
        id (int): 主キー
        timestamp (datetime): 診断実行時刻
        image_path (str): 診断対象画像のパス
        diagnosis (str): 診断結果（病名）
        confidence (float): 診断の確信度
        treatment (str): 推奨される治療法
        prevention (str): 予防策
        symptoms (str): 症状の説明
    """
    __tablename__ = 'diagnoses'

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    image_path = Column(String)
    diagnosis = Column(String)
    confidence = Column(Float)
    treatment = Column(Text)
    prevention = Column(Text)
    symptoms = Column(Text)


class Feedback(Base):
    """
    ユーザーからのフィードバックを保存するテーブル

    Attributes:
        id (int): 主キー
        diagnosis_id (int): 診断結果のID（外部キー）
        user_feedback (str): ユーザーからのフィードバック内容
        timestamp (datetime): フィードバック送信時刻
    """
    __tablename__ = 'feedbacks'

    id = Column(Integer, primary_key=True, index=True)
    diagnosis_id = Column(Integer)
    user_feedback = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
