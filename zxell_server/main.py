import json
import secrets
import shutil
from datetime import timedelta
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

import models
import schemas
import storage
from auth import get_current_client, require_admin
from config import settings
from database import Base, engine, get_db
from models import utcnow

Base.metadata.create_all(bind=engine)

app = FastAPI(title="zxell API Server")


# ---------- クライアント登録（承認制） ----------


@app.post("/api/clients/register", response_model=schemas.ClientRegisterOut)
def register_client(body: schemas.ClientRegisterIn, db: Session = Depends(get_db)):
    """登録申請。API キーは即時発行されるが、管理者が承認するまで使えない。"""
    api_key = secrets.token_urlsafe(32)
    client = models.Client(name=body.name, capabilities=body.capabilities, api_key=api_key)
    db.add(client)
    db.commit()
    return schemas.ClientRegisterOut(client_id=client.id, api_key=api_key, status=client.status)


@app.get(
    "/api/clients",
    response_model=List[schemas.ClientOut],
    dependencies=[Depends(require_admin)],
)
def list_clients(status: Optional[str] = None, db: Session = Depends(get_db)):
    """管理用: クライアント一覧。?status=pending で承認待ちのみ表示。"""
    stmt = select(models.Client).order_by(models.Client.created_at)
    if status:
        stmt = stmt.where(models.Client.status == status)
    return db.execute(stmt).scalars().all()


def _set_client_status(client_id: str, new_status: str, db: Session) -> models.Client:
    client = db.get(models.Client, client_id)
    if client is None:
        raise HTTPException(status_code=404, detail="Client not found")
    client.status = new_status
    db.commit()
    return client


@app.post(
    "/api/clients/{client_id}/approve",
    response_model=schemas.ClientOut,
    dependencies=[Depends(require_admin)],
)
def approve_client(client_id: str, db: Session = Depends(get_db)):
    """管理用: 登録申請を承認し、API キーを有効化する。"""
    return _set_client_status(client_id, "approved", db)


@app.post(
    "/api/clients/{client_id}/disable",
    response_model=schemas.ClientOut,
    dependencies=[Depends(require_admin)],
)
def disable_client(client_id: str, db: Session = Depends(get_db)):
    """管理用: クライアントを無効化する（却下・強制離脱にも使う）。"""
    return _set_client_status(client_id, "disabled", db)


# ---------- タスク ----------


def _requeue_expired(db: Session) -> None:
    """リース期限切れタスクを pending に戻し、試行超過分は failed にする。"""
    now = utcnow()
    db.execute(
        update(models.Task)
        .where(models.Task.status == "processing", models.Task.lease_expires_at < now)
        .values(
            status="pending",
            assigned_to=None,
            lease_expires_at=None,
            attempts=models.Task.attempts + 1,
            updated_at=now,
        )
    )
    db.execute(
        update(models.Task)
        .where(models.Task.status == "pending", models.Task.attempts >= models.Task.max_attempts)
        .values(status="failed", updated_at=now)
    )


@app.post(
    "/api/tasks",
    response_model=List[schemas.TaskOut],
    dependencies=[Depends(require_admin)],
)
def create_tasks(body: List[schemas.TaskCreateIn], db: Session = Depends(get_db)):
    """管理用: タスクの一括投入。"""
    created = []
    for item in body:
        if item.type not in models.TASK_TYPES:
            raise HTTPException(status_code=422, detail="unknown task type: %s" % item.type)
        task = models.Task(
            type=item.type,
            payload=item.payload,
            priority=item.priority,
            max_attempts=item.max_attempts,
        )
        db.add(task)
        created.append(task)
    db.commit()
    return created


@app.get("/api/tasks/next", response_model=schemas.TaskOut)
def next_task(
    types: Optional[str] = None,
    client: models.Client = Depends(get_current_client),
    db: Session = Depends(get_db),
):
    """リース付きタスク払い出し。types はカンマ区切りで希望タスク種別を絞り込める
    （例: GPU 機は types=train、CPU 機は types=preprocess,eval,verify）。"""
    _requeue_expired(db)

    stmt = (
        select(models.Task)
        .where(models.Task.status == "pending")
        .order_by(models.Task.priority.desc(), models.Task.id)
        .limit(1)
        # 二重払い出し防止（PostgreSQL の行ロック + SKIP LOCKED）
        .with_for_update(skip_locked=True)
    )
    if types:
        wanted = [t.strip() for t in types.split(",") if t.strip()]
        stmt = stmt.where(models.Task.type.in_(wanted))

    task = db.execute(stmt).scalars().first()
    if task is None:
        db.commit()
        raise HTTPException(status_code=404, detail="No pending tasks")

    task.status = "processing"
    task.assigned_to = client.id
    task.lease_expires_at = utcnow() + timedelta(seconds=settings.lease_seconds)
    task.updated_at = utcnow()
    db.commit()
    return task


# ---------- 結果提出 ----------


@app.post("/api/results", response_model=schemas.ResultOut)
def post_result(
    task_id: int = Form(...),
    base_weight_version: Optional[int] = Form(None),
    metrics: Optional[str] = Form(None),  # JSON 文字列
    artifact: Optional[UploadFile] = File(None),  # Δ（重み差分）等のバイナリ
    client: models.Client = Depends(get_current_client),
    db: Session = Depends(get_db),
):
    task = db.get(models.Task, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.assigned_to != client.id or task.status != "processing":
        raise HTTPException(status_code=409, detail="Task is not leased to this client")

    if task.type == "train" and base_weight_version is None:
        raise HTTPException(
            status_code=422, detail="train task requires base_weight_version"
        )
    if base_weight_version is not None:
        if db.get(models.WeightVersion, base_weight_version) is None:
            raise HTTPException(status_code=422, detail="unknown base_weight_version")

    metrics_dict = None
    if metrics:
        try:
            metrics_dict = json.loads(metrics)
        except ValueError:
            raise HTTPException(status_code=422, detail="metrics must be valid JSON")
        if not isinstance(metrics_dict, dict):
            raise HTTPException(status_code=422, detail="metrics must be a JSON object")

    artifact_path = None
    if artifact is not None:
        try:
            fname = storage.safe_name(artifact.filename or "artifact.bin")
        except ValueError:
            raise HTTPException(status_code=422, detail="invalid artifact filename")
        dest_dir = storage.results_dir() / ("task_%d" % task.id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / fname
        with dest.open("wb") as f:
            shutil.copyfileobj(artifact.file, f)
        artifact_path = str(dest.relative_to(settings.storage_dir))

    result = models.Result(
        task_id=task.id,
        client_id=client.id,
        base_weight_version=base_weight_version,
        metrics=metrics_dict,
        artifact_path=artifact_path,
    )
    task.status = "done"
    task.lease_expires_at = None
    task.updated_at = utcnow()
    db.add(result)
    db.commit()
    return result


# ---------- 重み配布 ----------


@app.post(
    "/api/weights",
    response_model=schemas.WeightVersionOut,
    dependencies=[Depends(require_admin)],
)
def upload_weights(
    file: UploadFile = File(...),
    note: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """管理用: 新しいグローバル重みスナップショットの登録。"""
    try:
        fname = storage.safe_name(file.filename or "weights.bin")
    except ValueError:
        raise HTTPException(status_code=422, detail="invalid filename")

    latest = db.execute(select(func.max(models.WeightVersion.version))).scalar()
    version = (latest or 0) + 1
    dest = storage.weights_dir() / ("v%06d_%s" % (version, fname))
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    wv = models.WeightVersion(
        version=version,
        filename=dest.name,
        checksum_sha256=storage.sha256_of(dest),
        size_bytes=dest.stat().st_size,
        note=note,
    )
    db.add(wv)
    db.commit()
    return wv


@app.get("/api/weights/latest", response_model=schemas.WeightVersionOut)
def latest_weights(
    client: models.Client = Depends(get_current_client), db: Session = Depends(get_db)
):
    wv = (
        db.execute(
            select(models.WeightVersion).order_by(models.WeightVersion.version.desc()).limit(1)
        )
        .scalars()
        .first()
    )
    if wv is None:
        raise HTTPException(status_code=404, detail="No weights registered")
    return wv


@app.get("/api/weights/{version}", response_model=schemas.WeightVersionOut)
def weight_metadata(
    version: int,
    client: models.Client = Depends(get_current_client),
    db: Session = Depends(get_db),
):
    wv = db.get(models.WeightVersion, version)
    if wv is None:
        raise HTTPException(status_code=404, detail="Unknown weight version")
    return wv


@app.get("/api/weights/{version}/download")
def download_weights(
    version: int,
    client: models.Client = Depends(get_current_client),
    db: Session = Depends(get_db),
):
    wv = db.get(models.WeightVersion, version)
    if wv is None:
        raise HTTPException(status_code=404, detail="Unknown weight version")
    path = storage.weights_dir() / wv.filename
    if not path.is_file():
        raise HTTPException(status_code=500, detail="Weight file missing on server")
    return FileResponse(path, filename=wv.filename)


# ---------- シャード配布 ----------


@app.get("/api/shards/{name}")
def download_shard(name: str, client: models.Client = Depends(get_current_client)):
    """タスク payload の shard 名を指定してトークン化済みシャードを取得する。"""
    try:
        fname = storage.safe_name(name)
    except ValueError:
        raise HTTPException(status_code=422, detail="invalid shard name")
    path = storage.shards_dir() / fname
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Shard not found")
    return FileResponse(path, filename=fname)


# ---------- ステータス（ダッシュボード用） ----------


@app.get(
    "/api/status",
    response_model=schemas.StatusOut,
    dependencies=[Depends(require_admin)],
)
def status(db: Session = Depends(get_db)):
    """管理用: タスク集計・クライアント一覧・最新重み（ダッシュボードのデータ源）。
    LAN 外からも到達しうるため X-Admin-Key 必須とした。"""
    _requeue_expired(db)
    db.commit()

    rows = db.execute(
        select(models.Task.type, models.Task.status, func.count())
        .group_by(models.Task.type, models.Task.status)
    ).all()
    tasks = {}
    for task_type, task_status, count in rows:
        tasks.setdefault(task_type, {})[task_status] = count

    clients = db.execute(select(models.Client)).scalars().all()
    latest = (
        db.execute(
            select(models.WeightVersion).order_by(models.WeightVersion.version.desc()).limit(1)
        )
        .scalars()
        .first()
    )
    return schemas.StatusOut(
        tasks=tasks,
        clients=[schemas.ClientOut.model_validate(c) for c in clients],
        latest_weight=schemas.WeightVersionOut.model_validate(latest) if latest else None,
    )
