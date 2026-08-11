import hmac

from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

import models
from config import settings
from database import get_db


def get_current_client(
    x_api_key: str = Header(...), db: Session = Depends(get_db)
) -> models.Client:
    """X-API-Key ヘッダで登録済みクライアントを認証し、last_seen を更新する。

    承認制: status が approved 以外（pending / disabled）は拒否する。
    """
    client = db.query(models.Client).filter(models.Client.api_key == x_api_key).first()
    if client is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if client.status != "approved":
        raise HTTPException(status_code=403, detail="Client not approved (status: %s)" % client.status)
    client.last_seen = models.utcnow()
    db.commit()
    return client


def require_admin(x_admin_key: str = Header(...)) -> None:
    """管理系 API（タスク投入・重み登録）用。ZXELL_ADMIN_API_KEY と照合する。"""
    if not hmac.compare_digest(x_admin_key, settings.admin_api_key):
        raise HTTPException(status_code=403, detail="Invalid admin key")
