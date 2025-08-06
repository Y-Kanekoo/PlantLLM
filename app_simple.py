import os
import io
import time
import asyncio
import hashlib
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from functools import lru_cache
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
from PIL import Image
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, Request, WebSocket, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv
from pydantic import BaseModel

# 環境変数の読み込み
load_dotenv()

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPIアプリケーションの初期化
app = FastAPI(
    title="Plant Disease Diagnosis System",
    description="植物の種類と病気の有無を診断するAIシステム",
    version="2.0.0"
)

# テンプレートとスタティックファイルの設定
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORSミドルウェアの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 基本設定
MAX_IMAGE_SIZE = 1024
ALLOWED_MIME_TYPES = {'image/jpeg', 'image/png', 'image/gif'}

# モデルの設定
MODELS_CONFIG = [
    {
        'name': 'gemini-2.0-flash-exp',
        'description': 'Gemini 2.0 Flash',
        'retry_count': 3,
        'wait_time': 4,
        'quota_limit': 60
    },
    {
        'name': 'gemini-1.5-flash',
        'description': 'Gemini 1.5 Flash',
        'retry_count': 3,
        'wait_time': 3,
        'quota_limit': 60
    },
    {
        'name': 'gemini-1.5-flash-8b',
        'description': 'Gemini 1.5 Flash-8B',
        'retry_count': 3,
        'wait_time': 2,
        'quota_limit': 60
    },
    {
        'name': 'gemini-1.5-pro',
        'description': 'Gemini 1.5 Pro',
        'retry_count': 3,
        'wait_time': 2,
        'quota_limit': 45
    }
]

# レート制限の統合管理


class UnifiedRateLimiter:
    def __init__(self):
        self.model_limiters = {}
        self.global_requests = []
        self.global_limit = 30  # グローバルな1分あたりの制限
        self.time_window = 60

    def initialize_model(self, model_name: str, quota_limit: int):
        """モデルごとのレート制限を初期化"""
        self.model_limiters[model_name] = {
            'quota_limit': quota_limit,
            'requests': [],
            'reset_time': None,
            'total_requests': 0,
            'error_count': 0
        }

    def is_allowed(self, model_name: str = None) -> bool:
        now = time.time()

        # グローバル制限のチェック
        self.global_requests = [
            req for req in self.global_requests if now - req < self.time_window]
        if len(self.global_requests) >= self.global_limit:
            logger.warning("Global rate limit reached")
            return False

        # モデル固有の制限をチェック
        if model_name and model_name in self.model_limiters:
            limiter = self.model_limiters[model_name]

            if limiter['reset_time'] and now < limiter['reset_time']:
                logger.warning(f"{model_name}: Still in cooldown period")
                return False

            limiter['requests'] = [req for req in limiter['requests']
                                   if now - req < self.time_window]
            if len(limiter['requests']) >= limiter['quota_limit']:
                limiter['reset_time'] = now + self.time_window
                logger.warning(
                    f"{model_name}: Model-specific rate limit reached")
                return False

            limiter['requests'].append(now)
            limiter['total_requests'] += 1

        self.global_requests.append(now)
        return True

    def record_error(self, model_name: str, error: Exception):
        """エラーを記録し、必要に応じてクールダウンを設定"""
        if model_name in self.model_limiters:
            limiter = self.model_limiters[model_name]
            limiter['error_count'] += 1

            if "429" in str(error):
                limiter['reset_time'] = time.time() + 60
                logger.warning(
                    f"{model_name}: API quota exceeded, cooling down")

    def get_stats(self, model_name: str = None):
        """使用統計を取得"""
        if model_name and model_name in self.model_limiters:
            limiter = self.model_limiters[model_name]
            return {
                'total_requests': limiter['total_requests'],
                'error_count': limiter['error_count'],
                'current_requests': len(limiter['requests']),
                'quota_limit': limiter['quota_limit'],
                'is_limited': bool(limiter['reset_time'] and time.time() < limiter['reset_time'])
            }
        return {
            'global_requests': len(self.global_requests),
            'global_limit': self.global_limit
        }


# 統合されたレート制限マネージャーの初期化
rate_manager = UnifiedRateLimiter()
for config in MODELS_CONFIG:
    rate_manager.initialize_model(config['name'], config['quota_limit'])

# 画像のハッシュ計算


def calculate_image_hash(image_bytes):
    return hashlib.md5(image_bytes).hexdigest()

# 診断結果のキャッシュ


@lru_cache(maxsize=100)
def get_cached_diagnosis(image_hash):
    return None  # キャッシュミスの場合はNone

# リトライ付きの画像処理


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=20)
)
async def process_image_with_retry(model, prompt, image):
    try:
        response = await model.generate_content_async([prompt, image])
        return response
    except Exception as e:
        logger.error(f"Error processing image with {model}: {str(e)}")
        if "429" in str(e):
            logger.warning(
                "API quota exceeded, waiting longer before retry...")
            await asyncio.sleep(10)
        raise


def process_image(image_data: bytes, mime_type: str) -> Dict[str, Any]:
    """画像の前処理を行う関数"""
    try:
        # 画像をPILで開く
        image = Image.open(io.BytesIO(image_data))

        # サイズの最適化
        if max(image.size) > MAX_IMAGE_SIZE:
            ratio = MAX_IMAGE_SIZE / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        # 画像をバイトに変換
        buffer = io.BytesIO()
        image.save(buffer, format=image.format or 'JPEG')
        processed_data = buffer.getvalue()

        return {
            'mime_type': mime_type,
            'data': processed_data
        }
    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}")
        raise ValueError(f"画像の処理に失敗しました: {str(e)}")


# Gemini APIの設定
try:
    # 新しいAPIキーを直接設定
    API_KEY = "..."
    genai.configure(api_key=API_KEY)
    logger.info("Successfully configured Gemini API with new key")
except Exception as e:
    logger.error(f"Failed to initialize Gemini API: {str(e)}")
    raise

# アップロードディレクトリの作成
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# 診断用のプロンプト
DIAGNOSIS_PROMPT = """
画像の植物を分析し、以下の3点について簡潔に回答してください：

1. 植物の種類：
2. 健康状態：[健康/病気の疑い]
3. 所見：

注意：
- 植物種が不明な場合は「不明」と記載
- 健康状態が判断できない場合は「判断不可」と記載
- 所見は観察された特徴を簡潔に記載
"""

# モデルの初期化


def initialize_model(model_config):
    try:
        model = genai.GenerativeModel(model_config['name'])
        logger.info(f"Initialized {model_config['description']}")
        return model
    except Exception as e:
        logger.warning(
            f"Failed to initialize {model_config['description']}: {str(e)}")
        return None


# 利用可能なモデルのリストを取得
available_models = []
for config in MODELS_CONFIG:
    model = initialize_model(config)
    if model:
        available_models.append({
            'model': model,
            'config': config
        })

if not available_models:
    logger.error("No models available")
    raise RuntimeError("利用可能なモデルがありません")

logger.info(f"Initialized {len(available_models)} models successfully")

# API使用状況のモニタリング


class APIMonitor:
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.last_error_time = None
        self.quota_reset_time = None

    def record_request(self):
        self.request_count += 1

    def record_error(self, error):
        self.error_count += 1
        self.last_error_time = time.time()
        if "429" in str(error):
            self.quota_reset_time = time.time() + 3600  # 1時間後にリセット想定


api_monitor = APIMonitor()

# モデルの状態を取得するエンドポイントを追加


@app.get("/model-status")
async def get_model_status():
    """各モデルの使用状況を返す"""
    return {
        model_name: limiter.get_stats()
        for model_name, limiter in rate_manager.model_limiters.items()
    }


async def process_image_with_models(prompt, image):
    last_error = None
    api_monitor.record_request()

    for model_info in available_models:
        model = model_info['model']
        config = model_info['config']

        # モデル固有のレート制限をチェック
        if not rate_manager.is_allowed(config['name']):
            logger.warning(
                f"Rate limit reached for {config['description']}, trying next model...")
            continue

        try:
            logger.info(f"Trying {config['description']}")
            response = await process_image_with_retry(model, prompt, image)
            logger.info(f"Success with {config['description']}")
            response.model_info = config['description']
            return response
        except Exception as e:
            last_error = e
            api_monitor.record_error(e)
            logger.warning(f"Failed with {config['description']}: {str(e)}")
            if "429" in str(e):
                rate_manager.record_error(config['name'], e)
            await asyncio.sleep(config['wait_time'])
            continue

    raise last_error or RuntimeError("全てのモデルでの処理に失敗しました")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """メインページを表示"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

# エラーメッセージの定義を更新
ERROR_MESSAGES = {
    'model_not_available': {
        'message': '申し訳ありません。現在このモデルは利用できません。',
        'action': '別のモデルを選択するか、しばらく待ってから再度お試しください。',
        'wait_time': 60,
        'severity': 'error'
    },
    'rate_limit': {
        'message': 'アクセスが集中しています。',
        'action': '1分ほど待ってから再度お試しください。',
        'wait_time': 60,
        'severity': 'warning'
    },
    'invalid_image': {
        'message': '画像の読み込みに失敗しました。',
        'action': '別の画像を試すか、画像を小さくしてお試しください。',
        'wait_time': 0,
        'severity': 'error'
    },
    'processing_error': {
        'message': '画像の処理中にエラーが発生しました。',
        'action': '別の画像で試すか, しばらく待ってから再度お試しください。',
        'wait_time': 30,
        'severity': 'error'
    },
    'network_error': {
        'message': 'ネットワークエラーが発生しました。',
        'action': 'インターネット接続を確認して、再度お試しください。',
        'wait_time': 10,
        'severity': 'warning'
    },
    'invalid_model': {
        'message': '指定されたモデルは現在利用できません。',
        'action': '別のモデルを選択してください。',
        'wait_time': 0,
        'severity': 'error'
    },
    'quota_exceeded': {
        'message': 'APIの利用制限に達しました。',
        'action': 'しばらく待ってから再度お試しください。',
        'wait_time': 60,
        'severity': 'warning'
    }
}


def get_user_friendly_error(error_type: str, original_error: str = None, model_name: str = None) -> dict:
    """ユーザーフレンドリーなエラーメッセージを生成"""
    error_info = ERROR_MESSAGES.get(error_type, {
        'message': '予期せぬエラーが発生しました。',
        'action': 'しばらく待ってから再度お試しください。',
        'wait_time': 30,
        'severity': 'error'
    })

    # モデル固有のレート制限情報を取得
    wait_time = error_info['wait_time']
    if error_type == 'rate_limit' and model_name:
        rate_limiter = rate_manager.model_limiters.get(model_name)
        if rate_limiter and rate_limiter['reset_time']:
            wait_time = max(
                0, int(rate_limiter['reset_time'] - time.time()))

    return {
        'error': True,
        'message': error_info['message'],
        'action': error_info['action'],
        'wait_time': wait_time,
        'severity': error_info['severity'],
        'model': model_name,
        'detail': str(original_error) if original_error else None,
        'timestamp': time.time()
    }


@app.post("/diagnose")
async def diagnose_plant(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    model_name: str = None
):
    """植物診断を実行"""
    try:
        start_time = time.time()

        # レート制限のチェック
        if not rate_manager.is_allowed(model_name):
            return JSONResponse(
                status_code=429,
                content=get_user_friendly_error(
                    'rate_limit', model_name=model_name)
            )

        # ファイルの検証
        if not file.content_type.startswith('image/'):
            return JSONResponse(
                status_code=400,
                content=get_user_friendly_error('invalid_image')
            )

        # 画像の読み込みと前処理
        try:
            contents = await file.read()
            image_hash = calculate_image_hash(contents)
            processed_image = process_image(contents, file.content_type)
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            return JSONResponse(
                status_code=400,
                content=get_user_friendly_error('processing_error', str(e))
            )

        # キャッシュされた診断結果をチェック
        cached_result = get_cached_diagnosis(image_hash)
        if cached_result:
            logger.info("Using cached diagnosis result")
            return JSONResponse(content=cached_result)

        # 診断の実行
        try:
            if model_name:
                selected_model = next(
                    (m for m in available_models if m['config']
                     ['name'] == model_name),
                    None
                )
                if not selected_model:
                    return JSONResponse(
                        status_code=400,
                        content=get_user_friendly_error('invalid_model')
                    )

                response = await process_image_with_retry(
                    selected_model['model'],
                    DIAGNOSIS_PROMPT,
                    processed_image
                )
                response.model_info = selected_model['config']['description']
            else:
                response = await process_image_with_models(DIAGNOSIS_PROMPT, processed_image)

            if not response:
                return JSONResponse(
                    status_code=500,
                    content=get_user_friendly_error('processing_error')
                )

            result = {
                "success": True,
                "diagnosis": response.text,
                "model_info": getattr(response, 'model_info', 'Unknown Model'),
                "processing_time": round(time.time() - start_time, 2),
                "timestamp": time.time(),
                "image_hash": image_hash
            }

            # 診断結果を履歴に追加
            try:
                logger.info("Adding diagnosis result to history...")
                entry = diagnosis_history.add_entry(result, contents)
                result["image_path"] = entry.get("image_path")
                logger.info(
                    f"Successfully added diagnosis to history. Entry ID: {entry['id']}")
            except Exception as e:
                logger.error(f"Failed to save diagnosis history: {str(e)}")
                # エラーは記録するが、診断結果は返す

            return JSONResponse(content=result)

        except Exception as e:
            error_type = 'quota_exceeded' if "429" in str(
                e) else 'model_not_available'
            if "429" in str(e):
                rate_manager.record_error(model_name, e)

            return JSONResponse(
                status_code=500,
                content=get_user_friendly_error(error_type, str(e), model_name)
            )

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=get_user_friendly_error('processing_error', str(e))
        )

# 診断履歴の管理


class DiagnosisHistory:
    def __init__(self):
        self.entries: List[DiagnosisEntry] = []
        self.next_id = 1
        self.load_history()

    def add_entry(self, image_path: str, diagnosis_result: Dict[str, Any],
                  model_used: str, processing_time: float) -> int:
        entry = DiagnosisEntry(
            id=self.next_id,
            timestamp=datetime.now().isoformat(),
            image_path=image_path,
            diagnosis_result=diagnosis_result,
            model_used=model_used,
            processing_time=processing_time
        )
        self.entries.append(entry)
        self.next_id += 1
        self.save_history()
        return entry.id

    def get_entry(self, entry_id: int) -> Optional[DiagnosisEntry]:
        for entry in self.entries:
            if entry.id == entry_id:
                return entry
        return None

    def add_chat_message(self, entry_id: int, role: str, message: str):
        entry = self.get_entry(entry_id)
        if entry:
            entry.chat_history.append({
                "role": role,
                "message": message,
                "timestamp": datetime.now().isoformat()
            })
            self.save_history()

    def get_history(self) -> List[DiagnosisEntry]:
        return self.entries

    def clear_history(self):
        self.entries = []
        self.save_history()

    def save_history(self):
        history_file = Path("history/diagnosis_history.json")
        history_file.parent.mkdir(exist_ok=True)
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump([entry.dict() for entry in self.entries], f,
                      ensure_ascii=False, indent=2)

    def load_history(self):
        history_file = Path("history/diagnosis_history.json")
        if history_file.exists():
            with open(history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.entries = [DiagnosisEntry(**entry) for entry in data]
                if self.entries:
                    self.next_id = max(entry.id for entry in self.entries) + 1


# 診断履歴マネージャーの初期化（グローバルスコープで初期化）
diagnosis_history = DiagnosisHistory()

# 履歴取得用のエンドポイントを追加


@app.get("/history")
async def get_diagnosis_history(limit: int = 10):
    """診断履歴を取得するエンドポイント"""
    return {
        "success": True,
        "history": diagnosis_history.get_history(limit)
    }

# 履歴削除用のエンドポイント


@app.post("/clear-history")
async def clear_diagnosis_history():
    """診断履歴を削除するエンドポイント"""
    try:
        diagnosis_history.clear_history()
        return JSONResponse(content={"success": True, "message": "履歴を削除しました"})
    except Exception as e:
        logger.error(f"Failed to clear history: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "履歴の削除に失敗しました"}
        )

# WebSocketコネクション管理


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        self.active_connections.pop(session_id, None)

    async def send_message(self, session_id: str, message: str):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_text(message)


manager = ConnectionManager()

# チャットエンドポイント（HTTP）


@app.post("/chat/{entry_id}")
async def chat_endpoint(entry_id: int, chat_message: ChatMessage):
    entry = diagnosis_history.get_entry(entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="診断エントリが見つかりません")

    # チャットモデルで応答を生成
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        chat = model.start_chat(history=[])
        response = await chat.send_message_async(chat_message.message)

        # チャット履歴に追加
        diagnosis_history.add_chat_message(
            entry_id, "user", chat_message.message)
        diagnosis_history.add_chat_message(
            entry_id, "assistant", response.text)

        return {
            "success": True,
            "session_id": chat_message.session_id,
            "reply": response.text
        }
    except Exception as e:
        logger.error(f"チャットエラー: {str(e)}")
        raise HTTPException(status_code=500, detail="チャット処理中にエラーが発生しました")

# WebSocketチャットエンドポイント


@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    try:
        while True:
            data = await websocket.receive_text()
            model = genai.GenerativeModel('gemini-1.5-pro')
            chat = model.start_chat(history=[])
            response = await chat.send_message_async(data)
            await manager.send_message(session_id, response.text)
    except Exception as e:
        logger.error(f"WebSocketエラー: {str(e)}")
    finally:
        manager.disconnect(session_id)

# 診断履歴の詳細取得エンドポイント


@app.get("/history/{entry_id}")
async def get_history_entry(entry_id: int):
    entry = diagnosis_history.get_entry(entry_id)
    if entry:
        return {"success": True, "entry": entry}
    return JSONResponse(
        status_code=404,
        content={"success": False, "message": "エントリが見つかりません"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "app_simple:app",
        host=os.getenv('HOST', '0.0.0.0'),
        port=int(os.getenv('PORT', 8000)),
        reload=True
    )
