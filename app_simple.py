import asyncio
import hashlib
import io
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
# selectinload は N+1 解消でサブクエリ方式に変更したため不要
from PIL import Image

from database import get_db, init_db
from models import ChatMessage, Diagnosis

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

APP_TITLE = "Plant Disease Diagnosis System"
APP_DESCRIPTION = "植物の種類と病気の有無を診断するAIシステム"
APP_VERSION = "3.0.0"

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
)

# CORS設定: 環境変数で許可オリジンを制御（本番では適切に制限すること）
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
FRONTEND_INDEX = FRONTEND_DIST / "index.html"
FRONTEND_ASSETS = FRONTEND_DIST / "assets"

if FRONTEND_ASSETS.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_ASSETS)), name="assets")

app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

MAX_IMAGE_SIZE = 1024
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
# image/jpgは非標準だが一部ブラウザ/クライアントが送信するため許容
ALLOWED_MIME_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/gif"}
MAX_HISTORY_LIMIT = 100  # /history の最大取得件数

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-2.0-flash-exp")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-1.5-pro")

MODELS_CONFIG = [
    {
        "name": "gemini-2.0-flash-exp",
        "description": "Gemini 2.0 Flash",
        "retry_count": 3,
        "wait_time": 4,
        "quota_limit": 60,
    },
    {
        "name": "gemini-1.5-flash",
        "description": "Gemini 1.5 Flash",
        "retry_count": 3,
        "wait_time": 3,
        "quota_limit": 60,
    },
    {
        "name": "gemini-1.5-flash-8b",
        "description": "Gemini 1.5 Flash-8B",
        "retry_count": 3,
        "wait_time": 2,
        "quota_limit": 60,
    },
    {
        "name": "gemini-1.5-pro",
        "description": "Gemini 1.5 Pro",
        "retry_count": 3,
        "wait_time": 2,
        "quota_limit": 45,
    },
]


class UnifiedRateLimiter:
    def __init__(self):
        self.model_limiters = {}
        self.global_requests: List[float] = []
        self.global_limit = 30
        self.time_window = 60

    def initialize_model(self, model_name: str, quota_limit: int) -> None:
        self.model_limiters[model_name] = {
            "quota_limit": quota_limit,
            "requests": [],
            "reset_time": None,
            "total_requests": 0,
            "error_count": 0,
        }

    def allow(self, model_name: str) -> bool:
        now = time.time()
        self.global_requests = [req for req in self.global_requests if now - req < self.time_window]
        if len(self.global_requests) >= self.global_limit:
            logger.warning("Global rate limit reached")
            return False

        limiter = self.model_limiters.get(model_name)
        if limiter:
            if limiter["reset_time"] and now < limiter["reset_time"]:
                logger.warning("%s: still in cooldown", model_name)
                return False

            limiter["requests"] = [req for req in limiter["requests"] if now - req < self.time_window]
            if len(limiter["requests"]) >= limiter["quota_limit"]:
                limiter["reset_time"] = now + self.time_window
                logger.warning("%s: model rate limit reached", model_name)
                return False

            limiter["requests"].append(now)
            limiter["total_requests"] += 1

        self.global_requests.append(now)
        return True

    def record_error(self, model_name: str, error: Exception) -> None:
        limiter = self.model_limiters.get(model_name)
        if not limiter:
            return
        limiter["error_count"] += 1
        if "429" in str(error):
            limiter["reset_time"] = time.time() + 60

    def get_stats(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        if model_name and model_name in self.model_limiters:
            limiter = self.model_limiters[model_name]
            return {
                "total_requests": limiter["total_requests"],
                "error_count": limiter["error_count"],
                "current_requests": len(limiter["requests"]),
                "quota_limit": limiter["quota_limit"],
                "is_limited": bool(
                    limiter["reset_time"] and time.time() < limiter["reset_time"]
                ),
            }
        return {
            "global_requests": len(self.global_requests),
            "global_limit": self.global_limit,
        }


rate_manager = UnifiedRateLimiter()
for config in MODELS_CONFIG:
    rate_manager.initialize_model(config["name"], config["quota_limit"])


def calculate_image_hash(image_bytes: bytes) -> str:
    return hashlib.md5(image_bytes).hexdigest()


def detect_mime_type(data: bytes) -> str:
    """ファイルのmagic bytesからMIMEタイプを検出する。"""
    # 各フォーマットのシグネチャ
    signatures = {
        b"\xff\xd8\xff": "image/jpeg",
        b"\x89PNG\r\n\x1a\n": "image/png",
        b"GIF87a": "image/gif",
        b"GIF89a": "image/gif",
    }
    for sig, mime in signatures.items():
        if data.startswith(sig):
            return mime
    return "application/octet-stream"


async def process_image_with_retry(
    client: genai.Client, model_name: str, prompt: str, image_data: bytes, retries: int = 3
) -> Any:
    """新しいgoogle.genai SDKを使用して画像診断を実行（リトライ付き）"""
    attempt = 0
    while attempt < retries:
        try:
            # 画像をPart.from_bytesで渡す
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=image_data, mime_type="image/jpeg"),
                ],
            )
            return response
        except Exception as exc:
            attempt += 1
            if attempt >= retries:
                raise
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                await asyncio.sleep(8)
            else:
                await asyncio.sleep(2)


def process_image(image_data: bytes) -> Dict[str, Any]:
    """画像をリサイズし、Gemini API用に処理する。RGBA/Pモードは自動でRGB変換。"""
    try:
        image = Image.open(io.BytesIO(image_data))

        # RGBA/Pモード（透過PNG等）はRGBに変換してJPEG保存エラーを防ぐ
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")

        max_side = max(image.size)
        if max_side > MAX_IMAGE_SIZE:
            ratio = MAX_IMAGE_SIZE / max_side
            new_size = tuple(int(dim * ratio) for dim in image.size)
            resample = (
                Image.Resampling.LANCZOS
                if hasattr(Image, "Resampling")
                else Image.LANCZOS
            )
            image = image.resize(new_size, resample)

        buffer = io.BytesIO()
        # 統一フォーマットでJPEG保存（品質85で圧縮）
        image.save(buffer, format="JPEG", quality=85)
        return {"mime_type": "image/jpeg", "data": buffer.getvalue()}
    except Exception as exc:
        logger.error("Image processing failed: %s", exc)
        raise ValueError(f"画像の処理に失敗しました: {exc}")


def get_extension_from_mime(mime_type: str) -> str:
    """MIMEタイプから拡張子を取得する（detect_mime_type基準）"""
    mime_map = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
    }
    return mime_map.get(mime_type, ".jpg")


def _cleanup_file(file_path: Path) -> None:
    """ファイルを安全に削除する（孤児ファイル対策用）"""
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info("Cleaned up orphan file: %s", file_path)
    except Exception as exc:
        logger.warning("Failed to cleanup file %s: %s", file_path, exc)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


# Google GenAI クライアント初期化（新SDK）
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai_client: Optional[genai.Client] = None

if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY is not set; diagnosis will fail")
else:
    try:
        genai_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Google GenAI client initialized successfully")
    except Exception as exc:
        logger.error("Failed to initialize GenAI client: %s", exc)


def get_available_models() -> List[Dict[str, Any]]:
    """利用可能なモデル設定を返す"""
    return [{"config": config} for config in MODELS_CONFIG]


available_models = get_available_models()
if not available_models:
    logger.error("No models configured")


def resolve_models(preferred: Optional[str]) -> List[Dict[str, Any]]:
    if preferred:
        for model_info in available_models:
            if model_info["config"]["name"] == preferred:
                return [model_info]
        return []

    default_first = []
    fallback = []
    for model_info in available_models:
        if model_info["config"]["name"] == DEFAULT_MODEL:
            default_first.append(model_info)
        else:
            fallback.append(model_info)
    return default_first + fallback


def format_diagnosis_response(entry: Diagnosis, used_cache: bool) -> Dict[str, Any]:
    return {
        "success": True,
        "entry_id": entry.id,
        "diagnosis": entry.diagnosis_text,
        "model": {
            "name": entry.model_name,
            "description": entry.model_description,
        },
        "processing_time": entry.processing_time,
        "timestamp": entry.created_at.isoformat(),
        "image": {
            "url": entry.image_path,
            "mime_type": entry.mime_type,
            "file_size": entry.file_size,
        },
        "from_cache": used_cache,
    }


DIAGNOSIS_PROMPT = """
画像の植物を観察し、以下の形式で日本語回答してください。

【植物の種類】
- 植物名:
- 学名(可能なら):
- 確信度(%):

【健康状態】
- 判定: 健康 / 病気の疑い / 判断不可
- 主な所見:
- 症状の深刻度: 低 / 中 / 高

【所見の詳細】
- 葉の特徴:
- 斑点・変色:
- 形状の変化:
- その他の観察:

【対処】
- すぐにできる対策:
- 予防のポイント:

注意:
- 画像が不鮮明な場合はその旨を記載
- 判断が難しい場合は複数候補を提示
- 確信度は必ず0-100で記載
"""


@app.on_event("startup")
async def on_startup() -> None:
    await init_db()


@app.get("/", include_in_schema=False)
async def serve_root() -> HTMLResponse:
    if FRONTEND_INDEX.exists():
        return FileResponse(FRONTEND_INDEX)
    return HTMLResponse(
        "<h1>PlantLLM</h1><p>frontend/dist が見つかりません。frontend をビルドしてください。</p>",
        status_code=200,
    )


@app.get("/model-status")
async def get_model_status() -> Dict[str, Any]:
    return {
        model_name: limiter.get_stats()
        for model_name, limiter in rate_manager.model_limiters.items()
    }


@app.get("/models")
async def get_models() -> Dict[str, Any]:
    return {
        "default": DEFAULT_MODEL,
        "models": [
            {
                "name": config["name"],
                "description": config["description"],
            }
            for config in MODELS_CONFIG
        ],
    }


class DiagnosisResult:
    """診断結果をラップするクラス（SDKレスポンスへの属性追加を避ける）"""
    def __init__(self, text: str, model_name: str, model_description: str):
        self.text = text
        self.model_name = model_name
        self.model_description = model_description


async def run_diagnosis(
    prompt: str, image_data: bytes, model_name: Optional[str]
) -> DiagnosisResult:
    """新しいgoogle.genai SDKを使用して診断を実行"""
    if not genai_client:
        raise HTTPException(status_code=500, detail="GenAIクライアントが初期化されていません")

    candidates = resolve_models(model_name)
    if model_name and not candidates:
        raise HTTPException(status_code=400, detail="指定されたモデルが見つかりません")
    if not candidates:
        raise HTTPException(status_code=500, detail="利用可能なモデルがありません")

    last_error = None
    for model_info in candidates:
        config = model_info["config"]
        if not rate_manager.allow(config["name"]):
            continue

        try:
            response = await process_image_with_retry(
                genai_client,
                config["name"],
                prompt,
                image_data,
                retries=config["retry_count"],
            )
            # ラッパークラスでモデル情報を返す（SDKレスポンスを変更しない）
            return DiagnosisResult(
                text=response.text,
                model_name=config["name"],
                model_description=config["description"],
            )
        except Exception as exc:
            last_error = exc
            rate_manager.record_error(config["name"], exc)
            await asyncio.sleep(config["wait_time"])
            continue

    if last_error:
        raise last_error
    raise HTTPException(status_code=429, detail="APIのレート制限に達しました")


@app.post("/diagnose")
async def diagnose_plant(
    file: UploadFile = File(...),
    model_name: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    start_time = time.time()

    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="対応していない画像形式です")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="ファイルサイズが大きすぎます")

    # サーバー側でMIMEタイプを検証（magic bytesチェック）
    detected_mime = detect_mime_type(contents)
    if detected_mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"画像形式の検証に失敗しました（検出: {detected_mime}）"
        )

    image_hash = calculate_image_hash(contents)

    # キャッシュ検索: image_hash + model_name の組み合わせで検索（画像保存前に実行）
    effective_model = model_name or DEFAULT_MODEL
    cached_query = await db.execute(
        select(Diagnosis)
        .where(
            Diagnosis.image_hash == image_hash,
            Diagnosis.model_name == effective_model,
        )
        .order_by(Diagnosis.created_at.desc())
        .limit(1)
    )
    cached = cached_query.scalar_one_or_none()

    used_cache = False
    diagnosis_text = None
    model_used = None
    model_desc = None
    image_path = None

    if cached:
        # キャッシュ命中: 既存の画像パスを再利用し、新規保存をスキップ
        used_cache = True
        diagnosis_text = cached.diagnosis_text
        model_used = cached.model_name
        model_desc = cached.model_description
        image_path = cached.image_path
    else:
        # キャッシュミス: 画像を保存して診断を実行
        # detect_mime_type基準で拡張子を決定（クライアントのcontent_typeではなく実際の画像形式）
        ext = get_extension_from_mime(detected_mime)
        filename = f"{uuid.uuid4().hex}{ext}"
        image_path = f"/uploads/{filename}"
        full_path = UPLOAD_DIR / filename
        with open(full_path, "wb") as buffer:
            buffer.write(contents)

        processed_image = process_image(contents)
        try:
            # 新SDKではimage_dataのbytesを直接渡す
            response = await run_diagnosis(DIAGNOSIS_PROMPT, processed_image["data"], model_name)
            if not response or not response.text:
                # 診断失敗時は保存したファイルを削除（孤児ファイル対策）
                _cleanup_file(full_path)
                raise HTTPException(status_code=500, detail="診断結果が空でした")
            diagnosis_text = response.text
            model_used = response.model_name
            model_desc = response.model_description
        except HTTPException:
            # HTTPExceptionの場合も保存したファイルを削除
            _cleanup_file(full_path)
            raise
        except Exception as exc:
            # 予期しないエラーの場合も保存したファイルを削除
            _cleanup_file(full_path)
            logger.error("Diagnosis failed: %s", exc)
            raise HTTPException(status_code=500, detail="診断に失敗しました")

    processing_time = round(time.time() - start_time, 2)

    entry = Diagnosis(
        created_at=datetime.utcnow(),
        image_path=image_path,
        image_hash=image_hash,
        mime_type=detected_mime,  # detect_mime_type基準で保存
        file_size=len(contents),
        diagnosis_text=diagnosis_text,
        model_name=model_used,
        model_description=model_desc,
        processing_time=processing_time,
    )
    db.add(entry)
    await db.commit()
    await db.refresh(entry)

    return JSONResponse(content=format_diagnosis_response(entry, used_cache))


@app.get("/history")
async def get_history(limit: int = 20, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    # 上限を超えないよう制限
    limit = min(max(1, limit), MAX_HISTORY_LIMIT)

    # N+1クエリ解消: サブクエリでchat_countを一括取得
    chat_count_subq = (
        select(
            ChatMessage.diagnosis_id,
            func.count(ChatMessage.id).label("chat_count"),
        )
        .group_by(ChatMessage.diagnosis_id)
        .subquery()
    )

    query = await db.execute(
        select(Diagnosis, chat_count_subq.c.chat_count)
        .outerjoin(chat_count_subq, Diagnosis.id == chat_count_subq.c.diagnosis_id)
        .order_by(Diagnosis.created_at.desc())
        .limit(limit)
    )
    rows = query.all()

    history = [
        {
            "id": entry.id,
            "timestamp": entry.created_at.isoformat(),
            "image": {
                "url": entry.image_path,
            },
            "diagnosis": entry.diagnosis_text,
            "model": {
                "name": entry.model_name,
                "description": entry.model_description,
            },
            "processing_time": entry.processing_time,
            "chat_count": chat_count or 0,
        }
        for entry, chat_count in rows
    ]

    return {"success": True, "history": history}


@app.get("/history/{entry_id}")
async def get_history_entry(entry_id: int, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    entry_query = await db.execute(select(Diagnosis).where(Diagnosis.id == entry_id))
    entry = entry_query.scalar_one_or_none()
    if not entry:
        raise HTTPException(status_code=404, detail="エントリが見つかりません")

    chat_query = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.diagnosis_id == entry_id)
        .order_by(ChatMessage.created_at.asc())
    )
    chats = [
        {
            "role": chat.role,
            "message": chat.message,
            "timestamp": chat.created_at.isoformat(),
        }
        for chat in chat_query.scalars().all()
    ]

    return {
        "success": True,
        "entry": {
            "id": entry.id,
            "timestamp": entry.created_at.isoformat(),
            "image": {
                "url": entry.image_path,
            },
            "diagnosis": entry.diagnosis_text,
            "model": {
                "name": entry.model_name,
                "description": entry.model_description,
            },
            "processing_time": entry.processing_time,
            "chat": chats,
        },
    }


@app.post("/clear-history")
async def clear_history(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    await db.execute(delete(ChatMessage))
    await db.execute(delete(Diagnosis))
    await db.commit()

    for file_path in UPLOAD_DIR.glob("*"):
        try:
            if file_path.is_file():
                file_path.unlink()
        except Exception as exc:
            logger.warning("Failed to remove %s: %s", file_path, exc)

    return {"success": True, "message": "履歴を削除しました"}


@app.post("/chat/{entry_id}")
async def chat_endpoint(
    entry_id: int,
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """新しいgoogle.genai SDKを使用したチャットエンドポイント"""
    if not genai_client:
        raise HTTPException(status_code=500, detail="GenAIクライアントが初期化されていません")

    entry_query = await db.execute(select(Diagnosis).where(Diagnosis.id == entry_id))
    entry = entry_query.scalar_one_or_none()
    if not entry:
        raise HTTPException(status_code=404, detail="診断エントリが見つかりません")

    chat_query = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.diagnosis_id == entry_id)
        .order_by(ChatMessage.created_at.asc())
    )
    history_rows = chat_query.scalars().all()

    # 新SDKでは履歴をtypes.Contentのリストとして渡す
    contents = []
    for msg in history_rows:
        role = "user" if msg.role == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg.message)]))

    # 新しいメッセージを追加
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=payload.message)]))

    # 診断結果をシステムコンテキストとして含める
    system_instruction = f"以下の植物診断結果に基づいて、ユーザーの質問に答えてください：\n\n{entry.diagnosis_text}"

    try:
        response = await genai_client.aio.models.generate_content(
            model=CHAT_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
            ),
        )
        reply_text = response.text
    except Exception as exc:
        logger.error("Chat failed: %s", exc)
        raise HTTPException(status_code=500, detail="チャットに失敗しました")

    db.add(ChatMessage(diagnosis_id=entry_id, role="user", message=payload.message))
    db.add(ChatMessage(diagnosis_id=entry_id, role="assistant", message=reply_text))
    await db.commit()

    return {"success": True, "reply": reply_text}


@app.get("/{full_path:path}", include_in_schema=False)
async def serve_spa(full_path: str) -> HTMLResponse:
    if FRONTEND_INDEX.exists():
        return FileResponse(FRONTEND_INDEX)
    raise HTTPException(status_code=404, detail="Not Found")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app_simple:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("RELOAD", "true").lower() == "true",
    )
