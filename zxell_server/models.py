import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    BigInteger,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
)

from database import Base


def utcnow():
    return datetime.now(timezone.utc)


def gen_client_id():
    return uuid.uuid4().hex


TASK_TYPES = ("preprocess", "train", "eval", "verify")
TASK_STATUSES = ("pending", "processing", "done", "failed")
CLIENT_STATUSES = ("pending", "approved", "disabled")


class Client(Base):
    """参加クライアント。登録時にサーバが API キーを発行する（承認制）。

    登録直後は status=pending で、管理者が /api/clients/{id}/approve で
    approved にするまで API キーは使えない。
    """

    __tablename__ = "clients"

    id = Column(String, primary_key=True, default=gen_client_id)
    name = Column(String, nullable=False)
    api_key = Column(String, unique=True, index=True, nullable=False)
    status = Column(String, nullable=False, default="pending", index=True)
    # GPU 種別・VRAM・ベンチ結果などの自己申告値。タスク種別と H の調整に使う
    capabilities = Column(JSON, nullable=True)
    trust_score = Column(Float, nullable=False, default=1.0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    last_seen = Column(DateTime(timezone=True), nullable=True)


class Task(Base):
    """タスク単位＝「シャード × ローカルステップ数 H」等。記事 1 件単位ではない。

    payload の例（type=train）:
        {"shard": "shard_00042.bin", "base_weight_version": 3, "local_steps": 200}
    """

    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True)
    type = Column(String, nullable=False, index=True)  # preprocess / train / eval / verify
    payload = Column(JSON, nullable=False, default=dict)
    status = Column(String, nullable=False, default="pending", index=True)
    priority = Column(Integer, nullable=False, default=0)
    assigned_to = Column(String, ForeignKey("clients.id"), nullable=True)
    # リース期限。期限切れは払い出し時に自動で pending に戻る（永久ロック防止）
    lease_expires_at = Column(DateTime(timezone=True), nullable=True)
    attempts = Column(Integer, nullable=False, default=0)
    max_attempts = Column(Integer, nullable=False, default=3)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow)


class Result(Base):
    """クライアントが提出した結果。Δ 等のバイナリはストレージに置き、パスだけ持つ。"""

    __tablename__ = "results"

    id = Column(Integer, primary_key=True)
    task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False, index=True)
    client_id = Column(String, ForeignKey("clients.id"), nullable=False, index=True)
    # ステイルネス判定に必須: どの重みバージョンを基準に計算した結果か
    base_weight_version = Column(Integer, ForeignKey("weight_versions.version"), nullable=True)
    metrics = Column(JSON, nullable=True)  # 損失・処理トークン数・処理時間など
    artifact_path = Column(String, nullable=True)  # storage_dir からの相対パス
    status = Column(String, nullable=False, default="received")  # received / verified / rejected
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class WeightVersion(Base):
    """グローバル重みのスナップショット管理。実体はストレージ内のファイル。"""

    __tablename__ = "weight_versions"

    version = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    checksum_sha256 = Column(String, nullable=False)
    size_bytes = Column(BigInteger, nullable=False)
    note = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
