import os
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import re
from difflib import get_close_matches
from utils import (
    load_image, get_dataset_labels, save_results_by_type,
    calculate_metrics, logger, model, normalize_path,
    safe_listdir, sample_images
)

BASE_DATASET_PATH = normalize_path("dataset/PlantVillage_Not_Divided")
OUTPUT_DIR = normalize_path("results/zero_shot")
BATCH_SIZE = 4  # GTX 1650 Tiのメモリを考慮
N_SAMPLES = 1000  # サンプリングする画像数


def create_prompt() -> str:
    """Zero-shot分類用のプロンプトを生成する"""
    return """この植物の画像を詳しく分析し、以下の情報を提供してください。
確信度の高い順に、最大5つの可能性のある診断結果を提示してください：

各診断は以下の形式で記載してください：
選択したラベル: [植物名 - 状態]
確信度: [0-100の数値]
特徴: [観察された特徴の簡潔な説明]

例：
選択したラベル: Apple - Black Rot
確信度: 85
特徴: 葉に茶色の病斑が見られ、Black Rotの典型的な症状を示している

選択したラベル: Apple - Apple Scab
確信度: 60
特徴: 一部に暗色の斑点も見られ、Apple Scabの可能性も考えられる

注意：
- 植物名と状態は「 - 」（ハイフン）で区切ってください
- 健康な場合は「[植物名] - Healthy」と記載してください
- 病気の場合は正確な病名を英語で記載してください"""


def extract_plant_info(response_text: str) -> tuple:
    """モデルの応答から植物の情報を抽出する"""
    plant_name = None
    condition = None

    for line in response_text.split('\n'):
        if '植物:' in line:
            plant_name = line.split(':')[1].strip()
        elif '状態:' in line:
            condition = line.split(':')[1].strip()

    return plant_name, condition


def map_to_dataset_label(plant_info: tuple, labels: list) -> str:
    """抽出した情報をデータセットのラベルにマッピングする"""
    plant_name, condition = plant_info

    # 最も近いラベルを探す
    potential_labels = [
        label for label in labels if plant_name.lower() in label.lower()]

    if not potential_labels:
        return None

    if condition.lower() == '健康' or '健康' in condition:
        health_labels = [
            label for label in potential_labels if 'healthy' in label.lower()]
        return health_labels[0] if health_labels else potential_labels[0]
    else:
        disease_labels = [
            label for label in potential_labels if 'healthy' not in label.lower()]
        if disease_labels:
            # 病名の類似度でマッピング
            return get_close_matches(condition, disease_labels, n=1, cutoff=0.1)[0]

    return potential_labels[0]


def process_single_image(args):
    """1枚の画像を処理する"""
    image_path, true_label, labels = args
    try:
        image = load_image(image_path)
        if image is None:
            return None

        # Geminiモデルによる分析
        response = model.generate_content([create_prompt(), image])
        response_text = response.text
        logger.debug(f"Model response: {response_text}")

        # 情報の抽出とラベルへのマッピング
        plant_info = extract_plant_info(response_text)
        predicted_label = map_to_dataset_label(plant_info, labels)

        return {
            'image_path': str(image_path),
            'true_label': true_label,
            'predicted_label': predicted_label,
            'raw_response': response_text
        }
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
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
    sampled_data = sample_images(image_data, n_samples=N_SAMPLES)
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

    # 評価指標の計算
    metrics = calculate_metrics(
        [r['predicted_label']
            for r in results if r['predicted_label'] is not None],
        [r['true_label'] for r in results if r['predicted_label'] is not None],
        [r['raw_response'] for r in results if r['predicted_label'] is not None]
    )

    logger.info(f"Completed processing dataset. "
                f"Accuracy: {metrics['accuracy']:.2%}, "
                f"Top-5 Accuracy: {metrics['top5_accuracy']:.2%}")

    return {
        'metrics': metrics,
        'detailed_results': results,
        'sampling_info': {
            'total_images': len(image_data),
            'sampled_images': len(sampled_data),
            'sampling_ratio': len(sampled_data) / len(image_data)
        }
    }


def main():
    """メイン処理"""
    # データセット全体を処理
    logger.info("Processing dataset...")
    results = process_dataset()
    logger.info("Completed processing dataset")

    # 結果の保存
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    save_results_by_type({'all': results}, OUTPUT_DIR, 'zero_shot')


if __name__ == "__main__":
    main()
