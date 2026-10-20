# -*- coding: utf-8 -*-
"""
密码哈希 + JWT 签发/校验（多端鉴权的根基，生产思维 #4）。

为什么这么做：
1. 密码不存明文——bcrypt 加盐哈希，数据库泄密也推不出原密码；
2. 为什么直接用 bcrypt 库而不是 passlib：passlib 1.7.4 已停维，与 bcrypt 4.x
   有已知兼容告警（每次运行打印吓人的 traceback），答辩演示不友好；bcrypt
   官方库 API 更简单且行为透明（requirements 里 passlib[bcrypt] 已带上 bcrypt）；
3. JWT（HS256）无状态：任何一台业务服务器都能验签，天然支持「多端登录数据一致」
   ——客户端带着 token 来，服务端从 token 里拿到 user_id，核心表全部按 user_id 隔离。
"""
import hashlib
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt  # PyJWT

from app.core.config import settings
from app.core.exceptions import AuthError

# JWT 载荷里放的键名（保持常量，避免各处写字符串拼错）
_CLAIM_USER_ID = "user_id"
_CLAIM_USERNAME = "username"


# ===== 服务化 API Key（M7-2）=====

# 前缀让 Key 在日志/配置里可一眼识别（与常见 sk- 前缀同理），不是安全边界
API_KEY_PREFIX = "zy_"
# 展示长度：前缀 + 8 位随机字符（够识别，远不足以推回完整 Key）
_API_KEY_PREFIX_LEN = len(API_KEY_PREFIX) + 8


def hash_api_key(plain: str) -> str:
    """API Key → sha256 十六进制摘要（定长 64）。

    为什么不是 bcrypt：Key 是高熵随机串（见 ApiKey 模型注释），无字典风险；
    sha256 可按唯一索引直查，而 bcrypt 只能全表逐个 verify。
    """
    return hashlib.sha256(plain.encode("utf-8")).hexdigest()


def generate_api_key() -> tuple[str, str, str]:
    """生成一把新 Key，返回 (明文, 展示前缀, 哈希)。

    明文只允许在创建响应里返回一次——调用方丢了只能吊销重发；
    256 位熵 + URL 安全字符集，可直接放 HTTP 头（X-API-Key）。
    """
    plain = API_KEY_PREFIX + secrets.token_urlsafe(32)
    return plain, plain[:_API_KEY_PREFIX_LEN], hash_api_key(plain)


def hash_password(password: str) -> str:
    """bcrypt 加盐哈希（自带随机盐，同一密码每次哈希不同——这是特性不是 bug）。"""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """校验明文密码与哈希是否匹配。"""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        # 哈希串本身损坏（如数据被改）→ 当作不匹配，不向上抛栈
        return False


def create_access_token(user_id: int, username: str) -> str:
    """签发访问令牌：载荷带 user_id/username，过期时间来自配置。

    为什么载荷里不放敏感信息：JWT 只是 base64 编码不加密，人人可读，
    所以只放「标识」，权限判断一律以数据库为准。
    """
    now = datetime.now(timezone.utc)
    payload = {
        _CLAIM_USER_ID: user_id,
        _CLAIM_USERNAME: username,
        "iat": now,  # 签发时间
        "exp": now + timedelta(minutes=settings.jwt_expire_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def decode_access_token(token: str) -> dict:
    """解析并校验令牌，返回载荷；任何异常都统一转成 AuthError（对外只说人话）。

    会拒绝：过期、签名被伪造（改过载荷）、乱码字符串。
    """
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    except jwt.ExpiredSignatureError as e:
        raise AuthError("登录已过期，请重新登录") from e
    except jwt.PyJWTError as e:
        # PyJWTError 覆盖签名错误/格式错误等全部解析失败场景
        raise AuthError("无效的登录凭证") from e

    if _CLAIM_USER_ID not in payload:
        raise AuthError("无效的登录凭证")
    return payload
