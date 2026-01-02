from utils import init_model
from utils import (
    load_image, get_dataset_labels, save_results_by_type,
    calculate_metrics, logger, normalize_path,
    safe_listdir, sample_images, calculate_plant_metrics, calculate_disease_metrics, extract_plant_name
)
import os
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import time

# モデル設定
MODEL_NAME = "models/gemini-2.0-flash-exp"  # Gemini 2.0 Flash

model = init_model(MODEL_NAME)

# 基本設定
BASE_DATASET_PATH = normalize_path(
    "dataset/PlantVillage_3Variants/color")  # 新しいデータセットのパスに変更
OUTPUT_DIR = normalize_path("results/closed_set")
BATCH_SIZE = 1  # 並列処理数を1に削減（APIの制限に対応）
N_SAMPLES = 50  # テスト用に50サンプルに増加
WAIT_TIME = 10  # リクエスト間隔を10秒に増加

# レート制限の定数
MAX_RPM = 10  # 1分あたりの最大リクエスト数
MAX_RPD = 1500  # 1日あたりの最大リクエスト数
SAFETY_FACTOR = 0.8  # 安全係数を80%に調整


def create_prompt() -> str:
    """Few-shot分類用のプロンプトを生成する"""
    return """植物の葉の画像を分析し、以下の形式で回答してください。
最も可能性の高い診断から順に、5つまで提示してください。

回答形式：
1番目の候補:
植物名: [植物名を英語で]
状態: [病名を英語で、または「Healthy」]
確信度: [0-100の数値]
特徴:
- 葉の特徴: [形状、色、テクスチャなど]
- 症状: [病変、変色などがある場合]

2番目の候補:
[同様の形式]

注意事項：
1. 植物名は以下のいずれかを使用：
   Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, 
   Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

2. 病名は以下のいずれかを使用（健康な場合は「Healthy」）：
   - Apple:
     * Apple scab: 暗褐色の斑点、葉が黄ばむ
     * Black rot: 茶色の円形病斑、紫色の縁取り
     * Cedar apple rust: オレンジ色の斑点、黄色いハロー
   - Cherry:
     * Powdery mildew: 白い粉状の被膜
   - Corn:
     * Cercospora leaf spot: 灰色～茶色の小さな斑点
     * Common rust: 赤褐色の隆起した斑点（pustules）
     * Northern Leaf Blight: 灰褐色の楕円形病斑、葉脈に沿って拡大
   - Grape:
     * Black rot: 茶色～黒色の不規則な斑点
     * Esca (Black Measles): 赤褐色の斑点、葉脈間が黄化
     * Leaf blight: 茶色の不規則な病斑
   - Orange:
     * Haunglongbing: 黄色の斑点、葉脈が黄化
   - Peach/Pepper:
     * Bacterial spot: 小さな茶色の斑点、黄色いハロー
   - Potato:
     * Early blight: 同心円状の褐色病斑
     * Late blight: 水浸状の暗色病斑、白いカビ
   - Squash:
     * Powdery mildew: 白い粉状の被膜
   - Strawberry:
     * Leaf scorch: 葉縁が紫～茶色に変色
   - Tomato:
     * Bacterial spot: 小さな黒色の斑点
     * Early blight: 同心円状の褐色病斑
     * Late blight: 水浸状の暗色病斑
     * Leaf Mold: 葉裏の白～灰色のカビ
     * Septoria leaf spot: 灰色の小斑点、暗褐色の縁
     * Spider mites: 葉の黄化、微細な斑点
     * Target Spot: 同心円状の大きな病斑
     * Mosaic virus: モザイク状の黄化
     * Yellow Leaf Curl Virus: 葉の黄化、巻き込み

3. 確信度は必ず0-100の数値で記載
4. 特徴は具体的に記載
5. 病気の診断では、上記の特徴的な症状との一致度を重視してください"""


def process_single_image(args):
    """1枚の画像を処理する

    Args:
        args (tuple): (image_path, true_label, labels)

    Returns:
        dict: 処理結果
    """
    image_path, true_label, labels = args
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            logger.debug(f"Processing image: {image_path}")
            start_time = time.time()

            image = load_image(image_path)
            if image is None:
                return None

            # Geminiモデルによる分類
            response = model.generate_content([create_prompt(), image])
            response_text = response.text
            logger.debug(f"Model response: {response_text}")

            # 応答から予測ラベルと確信度を抽出
            plant_name = None
            condition = None
            confidence = 0.0
            features = []

            for raw_line in response_text.split('\n'):
                line = raw_line.strip()
                if line.startswith('植物名:'):
                    plant_name = line.split(':', 1)[1].strip()
                elif line.startswith('状態:'):
                    condition = line.split(':', 1)[1].strip()
                elif line.startswith('確信度:'):
                    try:
                        confidence = float(line.split(':', 1)[1].strip().replace('%', ''))
                    except ValueError:
                        confidence = 0.0
                elif line.startswith('- '):
                    features.append(line[2:].strip())

            predicted_label = None
            if plant_name and condition:
                predicted_label = f"{plant_name} - {condition}"

            process_time = time.time() - start_time

            result = {
                'image_path': str(image_path),
                'true_label': true_label,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'features': '; '.join(features),
                'process_time': process_time,
                'raw_response': response_text
            }
            logger.debug(f"Processing result: {result}")

            # レート制限対応のための待機
            time.sleep(WAIT_TIME)
            return result

        except Exception as e:
            retry_count += 1
            error_msg = str(e)

            if "429" in error_msg:  # Rate limit error
                wait_time = WAIT_TIME * (2 ** retry_count)  # 指数バックオフ
                logger.warning(
                    f"Rate limit reached. Waiting {wait_time} seconds before retry {retry_count}/{max_retries}")
                time.sleep(wait_time)
            else:
                logger.error(f"Error processing {image_path}: {error_msg}")
                if retry_count < max_retries:
                    logger.info(
                        f"Retrying... Attempt {retry_count + 1}/{max_retries}")
                    time.sleep(WAIT_TIME)
                else:
                    logger.error(f"Max retries reached for {image_path}")
                    return None

    return None


def process_dataset() -> dict:
    """データセット全体を処理する

    Returns:
        dict: 処理結果
    """
    logger.info(f"Processing dataset at: {BASE_DATASET_PATH}")

    # ラベルリストの取得
    labels = get_dataset_labels(BASE_DATASET_PATH)
    logger.info(f"Found {len(labels)} labels in dataset")

    # 画像パスとラベルのリストを作成
    image_data = []
    for label in labels:
        dir_path = Path(BASE_DATASET_PATH) / label
        logger.debug(f"Processing directory: {dir_path}")

        if not dir_path.exists():
            logger.error(f"Directory does not exist: {dir_path}")
            continue

        for img_name in safe_listdir(dir_path):
            image_path = dir_path / img_name
            if image_path.is_file():
                image_data.append((image_path, label, labels))

    logger.info(f"Found {len(image_data)} total images")

    # ランダムサンプリング
    sampled_data = sample_images(image_data, n_samples_per_class=N_SAMPLES)
    logger.info(f"Sampled {len(sampled_data)} images for processing")

    # 並列処理で画像を分類
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        futures = [executor.submit(process_single_image, data)
                   for data in sampled_data]
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures),
                           desc="Processing images"):
            result = future.result()
            if result is not None:
                results.append(result)

    # 植物種の精度を計算
    plant_metrics = calculate_plant_metrics(
        [[extract_plant_name(r['predicted_label'])]
            for r in results if r['predicted_label'] is not None],
        [extract_plant_name(r['true_label'])
            for r in results if r['predicted_label'] is not None],
        [r['raw_response'] for r in results if r['predicted_label'] is not None]
    )

    # 病気の精度を計算
    disease_metrics = calculate_disease_metrics(
        [[r['predicted_label']]
            for r in results if r['predicted_label'] is not None],
        [r['true_label']
            for r in results if r['predicted_label'] is not None],
        [r['raw_response']
            for r in results if r['predicted_label'] is not None]
    )

    # 平均処理時間の計算
    avg_process_time = sum(r['process_time'] for r in results) / len(results)
    logger.info(
        f"Average processing time per image: {avg_process_time:.2f} seconds")

    logger.info(f"""Completed processing dataset.
    Plant Accuracy: {plant_metrics['accuracy']:.2%}
    Plant Top-5 Accuracy: {plant_metrics['top5_accuracy']:.2%}
    Disease Accuracy: {disease_metrics['accuracy']:.2%}
    Disease Top-5 Accuracy: {disease_metrics['top5_accuracy']:.2%}""")

    return {
        'plant_metrics': plant_metrics,
        'disease_metrics': disease_metrics,
        'detailed_results': results,
        'sampling_info': {
            'total_images': len(image_data),
            'sampled_images': len(sampled_data),
            'sampling_ratio': len(sampled_data) / len(image_data),
            'avg_process_time': avg_process_time
        }
    }


def main():
    """メイン処理"""
    logger.info("Starting closed-set classification")

    # データセット全体を処理
    results = process_dataset()

    # 結果の保存
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    save_results_by_type({'all': results}, OUTPUT_DIR, 'closed_set')

    logger.info("Completed all processing")


if __name__ == "__main__":
    main()
