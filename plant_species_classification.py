import os
import json
import random
from datetime import datetime
from pathlib import Path
import time
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import pandas as pd
import numpy as np
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# 環境変数の読み込み
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

"""
Gemini API制限情報（2024年1月現在）

1. Gemini 1.5 Flash
- 無料枠:
  - 15 リクエスト/分（RPM）
  - 100万 トークン/分（TPM）
  - 1,500 リクエスト/日（RPD）
- Pay-as-you-go:
  - 2,000 RPM
  - 400万 TPM

2. Gemini 2.0 Flash（試験運用版）
- 制限:
  - 10 RPM
  - 400万 TPM
  - 1,500 RPD

注意事項:
- APIキーの制限はプロジェクト単位で適用
- 新しいAPIキーを作成しても、同じプロジェクト内であれば制限は共有
- 制限をリセットするには新しいプロジェクトを作成する必要あり
"""

# モデルの設定
MODEL_NAME = "models/gemini-1.5-flash"  # Gemini 1.5 Flash

generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 1024,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    generation_config=generation_config,
    safety_settings=safety_settings
)


def load_and_prepare_image(image_path):
    """画像を読み込み、Gemini用に準備する"""
    try:
        image = Image.open(image_path)
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def handle_api_error(e, attempt, max_retries):
    """APIエラーを適切に処理する"""
    if "429" in str(e):  # Rate limit error
        backoff_time = min(30, 2 ** attempt * 5)
        print(f"Rate limit exceeded. Waiting {backoff_time} seconds...")
        return backoff_time
    elif "503" in str(e):  # Service unavailable
        print("Service temporarily unavailable. Waiting 10 seconds...")
        return 10
    elif "500" in str(e):  # Internal server error
        print("Internal server error. Waiting 5 seconds...")
        return 5
    return None


def get_plant_species(image_path, max_retries=3, retry_delay=5):
    """植物種を判別する"""
    image = load_and_prepare_image(image_path)
    if image is None:
        return None, None, 0

    prompt = """
    You are a botanical expert. This image shows plant leaves. 
    Please analyze the following characteristics in detail and provide your BEST guess even if you're not 100% certain.
    Always provide a plant identification with a confidence score.

    1. Detailed analysis of leaf morphology and structure:
       - Venation pattern (parallel, reticulate, etc.)
       - Leaf margin characteristics (serrate, entire, etc.)
       - Leaf surface features (glossy, presence of trichomes, etc.)
       - Leaf shape (linear, ovate, elliptical, etc.)
       - Leaf arrangement (alternate, opposite, whorled, etc.)

    2. Plant identification:
       - Most likely species (including scientific name)
       - Alternative possibilities (if any)
       - Family (including scientific name)
       - Confidence level of identification (0-100)
       - Reasoning for the identification

    3. Key identification features and distinguishing characteristics

    Please provide your response in the following JSON format:
    {
        "plant_name": {
            "common": "Most likely common name",
            "scientific": "Scientific name",
            "alternatives": ["Alternative 1", "Alternative 2"]
        },
        "family": {
            "common": "Family name",
            "scientific": "Family scientific name"
        },
        "confidence": Integer value from 0-100,
        "reasoning": "Detailed reasoning for the identification",
        "leaf_features": {
            "venation": "Venation pattern description",
            "margin": "Margin characteristics",
            "surface": "Surface features",
            "shape": "Leaf shape description",
            "arrangement": "Leaf arrangement pattern"
        },
        "identification_points": [
            "Key identification point 1",
            "Key identification point 2"
        ],
        "additional_notes": "Any additional relevant information"
    }

    Focus on providing accurate, scientific descriptions. Even if you're not completely certain,
    provide your best assessment with appropriate confidence level and reasoning.
    """

    for attempt in range(max_retries):
        try:
            print(
                f"Processing image: {image_path} (attempt {attempt + 1}/{max_retries})")
            start_time = time.time()
            response = model.generate_content([prompt, image])
            process_time = time.time() - start_time

            if not response.text:
                raise Exception("Empty response received")

            response_text = response.text
            json_str = response_text[response_text.find(
                "{"):response_text.rfind("}")+1]
            result = json.loads(json_str)
            result['process_time'] = process_time

            return result, response_text, process_time

        except Exception as e:
            print(f"Error in classification (attempt {attempt + 1}): {e}")
            backoff_time = handle_api_error(e, attempt, max_retries)

            if backoff_time and attempt < max_retries - 1:
                sleep(backoff_time)
                continue

            return None, None, 0

    print("Max retries exceeded")
    return None, None, 0


def process_image_batch(image_paths, batch_size=3):
    """
    画像をバッチで処理する（レート制限対応版）

    Args:
        image_paths (list): 処理する画像パスのリスト
        batch_size (int): バッチサイズ（デフォルト: 3）

    Returns:
        list: 処理結果のリスト
    """
    results = []
    total_images = len(image_paths)
    total_batches = (total_images + batch_size - 1) // batch_size
    processed_images = 0

    print(f"\n処理開始:")
    print(f"- 総画像数: {total_images}")
    print(f"- バッチサイズ: {batch_size}")
    print(f"- 総バッチ数: {total_batches}\n")

    for i in range(0, total_images, batch_size):
        current_batch = i // batch_size + 1
        batch = image_paths[i:i+batch_size]
        batch_size_actual = len(batch)

        print(f"\nバッチ {current_batch}/{total_batches} 処理中")
        print(f"- 画像数: {batch_size_actual}")
        print(f"- 進捗: {processed_images}/{total_images} 完了")

        batch_results = []
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = []
            for image_path in batch:
                futures.append(executor.submit(get_plant_species, image_path))

            for future in tqdm(futures, desc="画像処理"):
                try:
                    result = future.result()
                    if result[0]:  # result[0]はJSONデータ
                        batch_results.append(result)
                except Exception as e:
                    print(f"エラー発生: {e}")
                    continue

        results.extend(batch_results)
        processed_images += batch_size_actual

        if i + batch_size < total_images:
            wait_time = 15
            print(f"\nAPI制限対策: {wait_time}秒待機中...")
            sleep(wait_time)

    print(f"\n処理完了:")
    print(f"- 処理済み画像: {processed_images}/{total_images}")
    print(f"- 成功数: {len(results)}")

    return results


def evaluate_species_classification(dataset_path, num_samples=2, batch_size=2, save_results=True):
    """植物種の分類を評価する"""
    # 結果保存用のディレクトリを作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results") / f"analysis_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 健康な植物の種類を定義
    healthy_plants = [
        "Apple___healthy",
        "Blueberry___healthy",
        "Cherry_(including_sour)___healthy",
        "Corn_(maize)___healthy",
        "Grape___healthy",
        "Peach___healthy",
        "Pepper,_bell___healthy",
        "Potato___healthy",
        "Raspberry___healthy",
        "Soybean___healthy",
        "Strawberry___healthy",
        "Tomato___healthy"
    ]

    # 各植物種から画像を収集
    all_image_files = []
    for plant in healthy_plants:
        plant_path = Path(dataset_path) / plant
        if not plant_path.exists():
            print(f"Warning: Path not found - {plant_path}")
            continue

        # 大文字小文字を区別せずに画像ファイルを検索
        plant_images = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            plant_images.extend(list(plant_path.glob(f'*{ext}')))

        if not plant_images:
            print(f"Warning: No images found in {plant_path}")
            continue

        # 指定された数の画像をランダムに選択
        selected_images = random.sample(
            plant_images, min(num_samples, len(plant_images)))
        all_image_files.extend(selected_images)

    if not all_image_files:
        print(f"No images found in {dataset_path}")
        return None

    print(
        f"Processing {len(all_image_files)} images from {len(healthy_plants)} plant species...")

    # バッチ処理で画像を分類
    results = process_image_batch(all_image_files, batch_size=batch_size)

    # 分類結果を整理
    processed_results = []
    correct_predictions = 0
    total_predictions = 0

    for result in results:
        if result and result[0]:
            json_result = result[0]
            response_text = result[1]
            process_time = result[2]

            # 正解ラベルを取得（ディレクトリ名から）
            true_label = str(
                Path(all_image_files[total_predictions]).parent.name)
            json_result['true_label'] = true_label

            # 予測が正しいかチェック
            if json_result['plant_name']['common'].lower() in true_label.lower():
                correct_predictions += 1

            processed_results.append(json_result)
            total_predictions += 1

            # 結果を表示
            print("\n予測結果:")
            print(f"予測された植物: {json_result['plant_name']['common']}")
            print(f"信頼度: {json_result['confidence']}%")
            if 'alternatives' in json_result['plant_name']:
                print("代替候補:", ", ".join(
                    json_result['plant_name']['alternatives']))
            if 'reasoning' in json_result:
                print(f"判断根拠: {json_result['reasoning']}")
            print(f"正解ラベル: {true_label}")
            print(
                f"判定: {'✓ 正解' if json_result['plant_name']['common'].lower() in true_label.lower() else '× 不正解'}")

    # 全体の精度を計算
    accuracy = (correct_predictions / total_predictions *
                100) if total_predictions > 0 else 0
    print(
        f"\n全体の精度: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")

    if save_results:
        # 結果をJSONファイルとして保存
        results_data = {
            'accuracy': accuracy,
            'total_images': total_predictions,
            'correct_predictions': correct_predictions,
            'results': processed_results
        }

        json_path = results_dir / 'classification_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)

        print(f"Results saved to: {json_path}")

    return accuracy, processed_results


if __name__ == "__main__":
    try:
        print("\n植物種分類システム 実行開始")
        print("=" * 50)

        # データセットパスの設定と確認
        dataset_path = "dataset/PlantVillage_3Variants/color"
        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            raise FileNotFoundError(f"データセットが見つかりません: {dataset_path}")

        print(f"データセット: {dataset_path}")
        print(f"設定:")
        print(f"- サンプル数/種: 2")
        print(f"- バッチサイズ: 2")
        print("=" * 50)

        # 分類実行
        accuracy, results = evaluate_species_classification(
            dataset_path,
            num_samples=2,
            batch_size=2
        )

        if accuracy is not None:
            print("\n実行完了")
            print("=" * 50)
            print(f"最終精度: {accuracy:.2f}%")
            print(f"処理結果数: {len(results)}")
        else:
            print("\n警告: 結果が取得できませんでした")

    except FileNotFoundError as e:
        print(f"\nエラー: {str(e)}")
    except Exception as e:
        print(f"\n予期せぬエラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n処理を終了します")
