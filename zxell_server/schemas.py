from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

# ---------- クライアント ----------


class ClientRegisterIn(BaseModel):
    name: str
    # 例: {"gpu": "RTX 3060", "vram_gb": 12, "bench_tflops": 4.1, "upload_mbps": 30}
    capabilities: Optional[Dict[str, Any]] = None


class ClientRegisterOut(BaseModel):
    client_id: str
    api_key: str
    status: str  # 登録直後は pending。管理者承認後に approved となり API キーが有効になる


class ClientOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    status: str
    capabilities: Optional[Dict[str, Any]]
    trust_score: float
    last_seen: Optional[datetime]


# ---------- タスク ----------


class TaskCreateIn(BaseModel):
    type: str  # preprocess / train / eval / verify
    payload: Dict[str, Any] = Field(default_factory=dict)
    priority: int = 0
    max_attempts: int = 3


class TaskOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    type: str
    payload: Dict[str, Any]
    status: str
    priority: int
    attempts: int
    lease_expires_at: Optional[datetime]


# ---------- 結果 ----------


class ResultOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    task_id: int
    client_id: str
    base_weight_version: Optional[int]
    metrics: Optional[Dict[str, Any]]
    artifact_path: Optional[str]
    status: str
    created_at: datetime


# ---------- 重み ----------


class WeightVersionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    version: int
    filename: str
    checksum_sha256: str
    size_bytes: int
    note: Optional[str]
    created_at: datetime


# ---------- ステータス（ダッシュボード用） ----------


class StatusOut(BaseModel):
    tasks: Dict[str, Dict[str, int]]  # {タスク種別: {状態: 件数}}
    clients: List[ClientOut]
    latest_weight: Optional[WeightVersionOut]
