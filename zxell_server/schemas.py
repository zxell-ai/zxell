from pydantic import BaseModel
from typing import Optional

class TrainingDataOut(BaseModel):
    id: int
    content: str
    title: Optional[str]
    summary: Optional[str]
    category: Optional[str]

    class Config:
        orm_mode = True

class ResultIn(BaseModel):
    data_id: int
    client_id: str
    result_json: dict

