from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, text
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from sqlalchemy.dialects.postgresql import UUID
from app.db.database import Base


class RawTranscript(Base):
    __tablename__ = "raw_transcripts"

    id = Column(Integer, primary_key=True, index=True)
    uid = Column(
        UUID(as_uuid=True),
        server_default=text("gen_random_uuid()"),
        unique=True,
        nullable=False,
    )
    session_id = Column(
        Integer,
        ForeignKey("counseling_sessions.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    total_segments = Column(Integer, nullable=False)
    raw_transcript = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # One-to-one relationship with CounselingSession
    session = relationship(
        "CounselingSession", uselist=False, back_populates="raw_transcript"
    )
