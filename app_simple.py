from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import logging
import google.generativeai as genai
from dotenv import load_dotenv
import os
import uvicorn
from PIL import Image
import io
from typing import Dict, Any
import time

# 環境変数の読み込み
load_dotenv()

# ロギングの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# FastAPIアプリケーションの初期化
app = FastAPI(
    title="Simple Plant Disease Diagnosis",
    description="植物の種類と病気の有無を診断するシンプルなAIシステム",
    version="1.0.0"
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

# 定数の定義
MAX_IMAGE_SIZE = 1024  # 最大画像サイズ
ALLOWED_MIME_TYPES = {'image/jpeg', 'image/png', 'image/gif'}  # 許可する画像形式


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
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    model = genai.GenerativeModel('models/gemini-2.0-flash-exp')
    logger.info("Gemini API initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini API: {str(e)}")
    raise

# アップロードディレクトリの作成
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """メインページを表示"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post("/diagnose")
async def diagnose_plant(file: UploadFile = File(...)):
    """シンプルな植物診断を実行"""
    start_time = time.time()  # 処理開始時間を記録

    try:
        # MIMEタイプの検証
        if file.content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=400,
                detail="サポートされていない画像形式です。JPEG、PNG、GIF形式のみ対応しています。"
            )

        # ファイルの読み込みと保存
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:  # 10MB制限
            raise HTTPException(
                status_code=400,
                detail="ファイルサイズが大きすぎます。10MB以下にしてください。"
            )

        # 画像の前処理
        processed_image = process_image(content, file.content_type)

        # Geminiモデルによる診断
        prompt = """
        この植物の画像を分析し、以下の2点のみについて簡潔に回答してください：

        1. この植物の種類：
        2. 病気の有無：[あり/なし]

        注意：
        - 植物の種類が不明な場合は「不明」と記載してください
        - 病気の有無が判断できない場合は「判断不可」と記載してください
        """

        # 診断の実行
        response = model.generate_content([prompt, processed_image])
        if not response.text:
            raise ValueError("診断結果が空でした")

        # 処理時間の計算
        process_time = time.time() - start_time

        # 診断結果の作成
        diagnosis_with_time = f"""
{response.text}

==============================
診断処理時間: {process_time:.2f}秒
"""

        return JSONResponse(content={
            "status": "success",
            "diagnosis": diagnosis_with_time
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"リクエストの処理に失敗しました: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "app_simple:app",
        host=os.getenv('HOST', '0.0.0.0'),
        port=int(os.getenv('PORT', 8000)),
        reload=True
    )
