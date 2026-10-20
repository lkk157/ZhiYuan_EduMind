# -*- coding: utf-8 -*-
"""
API Key 管理接口（M7-2）：用 JWT 登录的人给自己签发/查看/吊销机器调用凭证。

为什么管理接口用 JWT 而不是 API Key：
「签发 Key 的权限」不能由 Key 本身授予——否则一把泄露的 Key 可以自签无穷多把；
管理动作只有人（登录态）能做，调用动作才交给 Key（/v1）。

为什么不做前端页面：服务化的调用方是其他系统/运维脚本，
curl 一次即可完成签发；为它加一个 Streamlit 页面收益低、攻击面反而变大。
"""
from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field

from app.api.auth import get_current_user
from app.core.exceptions import NotFoundError
from app.db import crud
from app.db.models import User
from app.db.session import get_db

router = APIRouter(prefix="/api-keys", tags=["api-keys"])


class CreateKeyRequest(BaseModel):
    """签发入参：name 是调用方名称（给管理员辨认用，不影响鉴权）。"""

    name: str = Field(min_length=1, max_length=64, description="调用方名称，如「课程门户」")


@router.post("", status_code=status.HTTP_201_CREATED)
def create_key(
    body: CreateKeyRequest,
    user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """签发一把 Key：响应里的 key 字段**只出现这一次**，之后只能看到前缀。"""
    row, plain = crud.create_api_key(db, user_id=user.id, name=body.name)
    return {
        "id": row.id,
        "name": row.name,
        "key": plain,  # ★ 仅此一次；数据库只存 sha256 哈希
        "key_prefix": row.key_prefix,
        "created_at": row.created_at,
    }


@router.get("")
def list_keys(user: User = Depends(get_current_user), db=Depends(get_db)):
    """本人全部 Key（含已吊销）：不含明文、不含哈希，只给前缀与状态。"""
    return [
        {
            "id": k.id,
            "name": k.name,
            "key_prefix": k.key_prefix,
            "is_active": k.is_active,
            "last_used_at": k.last_used_at,
            "created_at": k.created_at,
        }
        for k in crud.list_api_keys(db, user_id=user.id)
    ]


@router.delete("/{kid}")
def revoke_key(
    kid: int,
    user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """吊销一把 Key（软删保留审计；非本人/不存在统一 404 防探测）。"""
    if not crud.revoke_api_key(db, user_id=user.id, key_id=kid):
        raise NotFoundError("API Key 不存在")
    return {"ok": True}
