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
import time  # 時間計測用に追加

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
    title="Plant Disease Diagnosis System",
    description="植物の病気を診断するAIシステム",
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

# 植物リストの定義
PLANT_LIST = {
    "トマト": {
        "scientific_name": "Solanum lycopersicum",
        "common_diseases": ["葉カビ病", "細菌性斑点病", "早期枯病", "遅延枯病", "モザイクウイルス", "黄化葉巻病", "ターゲット斑点病", "すすかび病"]
    },
    "ジャガイモ": {
        "scientific_name": "Solanum tuberosum",
        "common_diseases": ["早期枯病", "遅延枯病"]
    },
    "リンゴ": {
        "scientific_name": "Malus domestica",
        "common_diseases": ["黒星病", "黒腐病", "さび病"]
    },
    "ブドウ": {
        "scientific_name": "Vitis vinifera",
        "common_diseases": ["べと病", "黒とう病", "さび病", "黒点病"]
    },
    "トウモロコシ": {
        "scientific_name": "Zea mays",
        "common_diseases": ["ごま葉枯病", "さび病", "すす紋病"]
    },
    "ピーマン": {
        "scientific_name": "Capsicum annuum",
        "common_diseases": ["細菌性斑点病"]
    },
    "オレンジ": {
        "scientific_name": "Citrus × sinensis",
        "common_diseases": ["カンキツグリーニング病"]
    },
    "ブルーベリー": {
        "scientific_name": "Vaccinium corymbosum",
        "common_diseases": []
    },
    "イチゴ": {
        "scientific_name": "Fragaria × ananassa",
        "common_diseases": ["葉焼病"]
    },
    "サクランボ": {
        "scientific_name": "Prunus avium",
        "common_diseases": ["うどんこ病"]
    },
    "モモ": {
        "scientific_name": "Prunus persica",
        "common_diseases": ["細菌性斑点病"]
    },
    "ラズベリー": {
        "scientific_name": "Rubus idaeus",
        "common_diseases": []
    },
    "カボチャ": {
        "scientific_name": "Cucurbita",
        "common_diseases": ["うどんこ病"]
    },
    "大豆": {
        "scientific_name": "Glycine max",
        "common_diseases": []
    }
}


def process_image(image_data: bytes, mime_type: str) -> Dict[str, Any]:
    """
    画像の前処理を行う関数

    Args:
        image_data (bytes): 画像データ
        mime_type (str): MIMEタイプ

    Returns:
        Dict[str, Any]: 処理済み画像データ
    """
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
    logger.info(
        "Gemini API initialized successfully with models/gemini-2.0-flash-exp model")
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
    """
    植物の画像をアップロードして病気を診断する

    Args:
        file (UploadFile): アップロードされた画像ファイル

    Returns:
        dict: 診断結果
    """
    start_time = time.time()  # 処理開始時間を記録
    try:
        # MIMEタイプの検証
        if file.content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=400,
                detail="サポートされていない画像形式です。JPEG、PNG、GIF形式のみ対応しています。"
            )

        logger.info(f"Received file: {file.filename}")
        logger.debug(f"Content type: {file.content_type}")

        # ファイルの保存
        file_path = UPLOAD_DIR / file.filename
        content = await file.read()

        # ファイルサイズの検証
        if len(content) > 10 * 1024 * 1024:  # 10MB制限
            raise HTTPException(
                status_code=400,
                detail="ファイルサイズが大きすぎます。10MB以下にしてください。"
            )

        with open(file_path, "wb") as buffer:
            buffer.write(content)
        logger.debug(f"File saved at: {file_path}, size: {len(content)} bytes")

        # 画像の診断
        try:
            # 画像の前処理
            processed_image = process_image(content, file.content_type)
            logger.debug(
                f"Image processed successfully: size={len(processed_image['data'])} bytes")

            # Geminiモデルによる診断
            prompt = f"""
            以下の手順で植物の画像を分析し、各セクションを明確に分けて日本語で提供してください。
            対応している植物の種類は以下の通りです：

            対応植物リスト：
            {', '.join(PLANT_LIST.keys())}

            【植物の基本情報】
            - 植物の種類（上記リストから選択、確信度%を併記）：
            - 学名：
            - 全体的な状態：
            - 特徴的な外観：

            【植物種の候補（Top-5）】
            1位: [植物名] (確信度%)
            2位: [植物名] (確信度%)
            3位: [植物名] (確信度%)
            4位: [植物名] (確信度%)
            5位: [植物名] (確信度%)

            【健康状態の診断】
            - 病気の有無：
            - 症状の具体的な状態：
            - 深刻度：[軽度/中度/重度]
            - 診断の確信度（%）：

            【詳細な症状分析】
            ■ 各部位の状態
            - 葉：
            - 茎：
            - 花（ある場合）：
            - 実（ある場合）：

            ■ 症状の詳細
            - 変色：
            - 斑点：
            - 萎れ：
            - 病気の進行状態：

            【病気の種類】
            - 最も可能性の高い病気（確信度%）：
            - 他の可能性のある病気（Top-4、確信度%）：
              1. 
              2. 
              3. 
              4. 

            【推奨される対処方法】
            ■ 即時的な対応策：

            ■ 長期的な予防策：

            ■ 栽培環境の調整：

            ■ 専門家への相談：[必要/不要]

            注意事項：
            - 画像が不鮮明な場合や判断が難しい場合は、その旨を明記してください
            - 確信が持てない場合は、可能性のある複数の診断を提示してください
            - 深刻な症状の場合は、専門家への相談を推奨してください
            - 植物の種類の判別が難しい場合は、似ている可能性のある植物を全て挙げてください
            - 必ず確信度をパーセンテージで示してください
            """

            logger.debug("Sending request to Gemini API...")
            response = model.generate_content([prompt, processed_image])
            logger.debug(f"Received response from Gemini API: {response}")

            if not response.text:
                raise ValueError("診断結果が空でした")

            diagnosis = response.text
            process_time = time.time() - start_time  # 処理時間を計算

            # 処理時間を診断結果に追加
            diagnosis_with_time = f"""
            {diagnosis}

            ==============================
            【精度情報】
            植物種の識別：
            - 通常精度: {metrics['plant_accuracy']}%
            - Top-5精度: {metrics['plant_top5_accuracy']}%

            病気の診断：
            - 通常精度: {metrics['disease_accuracy']}%
            - Top-5精度: {metrics['disease_top5_accuracy']}%

            ==============================
            診断処理時間: {process_time:.2f}秒
            """

            logger.info("Diagnosis completed successfully")
            logger.debug(f"Diagnosis result: {diagnosis_with_time}")

            return JSONResponse(content={
                "status": "success",
                "diagnosis": diagnosis_with_time,
                "metrics": {
                    "plant_metrics": {
                        "accuracy": metrics['plant_accuracy'],
                        "top5_accuracy": metrics['plant_top5_accuracy']
                    },
                    "disease_metrics": {
                        "accuracy": metrics['disease_accuracy'],
                        "top5_accuracy": metrics['disease_top5_accuracy']
                    }
                },
                "metadata": {
                    "file_name": file.filename,
                    "file_size": len(content),
                    "processed_size": len(processed_image['data']),
                    "mime_type": file.content_type,
                    "timestamp": str(Path(file_path).stat().st_mtime)
                }
            })

        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Diagnosis failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"画像の診断に失敗しました: {str(e)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"リクエストの処理に失敗しました: {str(e)}"
        )

if __name__ == "__main__":
    # サーバーの起動
    uvicorn.run(
        "app:app",
        host=os.getenv('HOST', '0.0.0.0'),
        port=int(os.getenv('PORT', 8000)),
        reload=os.getenv('RELOAD', 'True').lower() == 'true'
    )
