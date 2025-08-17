from pydantic import BaseModel, EmailStr
from typing import Optional
from uuid import UUID


# Schema for creating a new counselor
class CounselorCreate(BaseModel):
    name: str
    employee_id: str
    dept: str
    email: EmailStr
    mobile_number: str


# Schema for updating an existing counselor
class CounselorUpdate(BaseModel):
    name: Optional[str] = None
    employee_id: Optional[str] = None
    dept: Optional[str] = None
    email: Optional[EmailStr] = None
    mobile_number: Optional[str] = None


# Schema for reading counselor data (without internal integer id)
class CounselorResponse(BaseModel):
    uid: UUID
    name: str
    employee_id: str
    dept: str
    email: EmailStr
    mobile_number: str

    model_config = {"from_attributes": True}


# Paginated list response for counselors
from typing import List


class CounselorListResponse(BaseModel):
    items: List[CounselorResponse]
    total: int
    skip: int
    limit: int
