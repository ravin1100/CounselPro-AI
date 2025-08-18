from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, HttpUrl, Field


# Base schema (shared fields)
class SessionBase(BaseModel):
    counselor_uid: str
    description: str
    session_date: datetime
    recording_link: HttpUrl


# Create schema
class SessionCreate(SessionBase):
    pass


# Update schema
class SessionUpdate(BaseModel):
    counselor_uid: str | None = None
    session_date: datetime | None = None
    recording_link: HttpUrl | None = None


# Counselor sub-schema (for response only)
class CounselorInfo(BaseModel):
    uid: UUID
    name: str

    model_config = {"from_attributes": True}


# Response schema


class SessionResponse(BaseModel):
    uid: UUID
    description: str
    session_date: datetime
    recording_link: HttpUrl
    counselor: CounselorInfo  # nested counselor info


# Paginated list response for sessions
from typing import List


class SessionListResponse(BaseModel):
    items: List[SessionResponse]
    total: int
    skip: int
    limit: int

    model_config = {"from_attributes": True}
