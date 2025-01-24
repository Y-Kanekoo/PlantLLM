import os
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import re
import json
from datetime import datetime
from difflib import get_close_matches
from utils import (
    load_image, get_dataset_labels, save_results_by_type,
    calculate_metrics, logger, normalize_path,
    safe_listdir, sample_images, calculate_plant_metrics,
    calculate_disease_metrics
)
import google.generativeai as genai
from time import sleep

# 基本設定
BASE_DATASET_PATH = normalize_path("dataset/PlantVillage_3Variants/color")
OUTPUT_DIR = normalize_path("results/zero_shot")
BATCH_SIZE = 2  # Gemini 2.0 Flash の制限に対応
N_SAMPLES = 1000  # サンプル数を1000に増やす
WAIT_TIME = 10  # API制限に対応するため10秒のインターバルを維持

# Gemini 2.0 Flash の設定
MODEL_NAME = "models/gemini-2.0-flash-exp"
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel(MODEL_NAME)


def create_prompt() -> str:
    """Zero-shot分類用のプロンプトを生成する"""
    return """この植物の葉の画像を、家庭菜園をする人の目線で詳しく観察し、分析結果を提示してください。
最も可能性の高い診断から順に、5つまで提示してください。

観察のポイント：
1. 葉の基本的な特徴
   - 形状（丸い、細長い、ギザギザなど）
   - サイズ（大きい、小さい、普通）
   - 表面の様子（つるつる、毛がある、ざらざら）
   - 厚み（厚い、薄い、普通）
   - 葉脈の特徴（はっきりしている、目立たない）

2. 健康状態の観察
   - 全体的な色（濃い緑、薄い緑、黄色みがかっているなど）
   - 変色の有無と特徴（斑点、まだら、縁の変色など）
   - 乾燥や萎れの状態
   - 虫食いや傷の有無

3. 病気の可能性がある場合の詳細観察
   - 症状の現れ方（上の葉から、下の葉から、まばらなど）
   - 斑点や病変の特徴（色、形、大きさ、広がり方）
   - 周囲の変化（黄色い輪、乾燥、変形など）

回答形式：
1番目の候補:
植物名: [観察された特徴から推測される植物の名前を英語で]
状態: [観察された症状や健康状態を英語で]
確信度: [0-100の数値]
特徴:
- 観察された特徴:
  * 葉の特徴（形、大きさ、表面、厚み、葉脈）
  * 全体的な印象（色、つや、健康状態）
- 気になる症状:
  * どこに（葉のどの部分に症状があるか）
  * どのように（症状の現れ方、進行状況）
  * 特徴的な点（他の病気と区別できる特徴）
- 判断の理由:
  * なぜそう考えたか
  * 似ている症状との違い
  * 深刻さの程度

2番目の候補:
[同様の形式]

確信度の判断基準：
90-100%: すべての特徴が一致し、他の可能性をほぼ排除できる
70-89%: 主要な特徴が一致し、他の可能性は低い
50-69%: 特徴の一部が一致するが、他の可能性も考えられる
30-49%: 特徴が曖昧で、複数の可能性がある
0-29%: 判断材料が不足している、または特徴が一致しない

注意事項：
1. 見たままの特徴を客観的に記載する
2. 不確かな点は正直に「不明」と記載する
3. 深刻度は症状の広がりと程度から判断する
4. 予防的な処置が必要な場合は明記する"""


def extract_predictions(response_text: str) -> list:
    """モデルの応答から予測を抽出する"""
    predictions = []
    current_prediction = {}

    logger.debug(f"Raw response text:\n{response_text}")

    for line in response_text.split('\n'):
        line = line.strip()

        if '植物名:' in line:
            if current_prediction:
                predictions.append(current_prediction)
                current_prediction = {}
            plant_name = line.split(':')[1].strip()
            current_prediction['plant'] = plant_name

        elif '状態:' in line:
            disease = line.split(':')[1].strip()
            current_prediction['disease'] = disease

        elif '確信度:' in line:
            try:
                confidence = float(line.split(':')[1].strip().replace('%', ''))
                current_prediction['confidence'] = confidence
            except ValueError:
                current_prediction['confidence'] = 0.0

        elif '特徴:' in line or '葉の特徴:' in line or '症状:' in line:
            if '葉の特徴:' in line:
                current_prediction['leaf_features'] = line.split(':')[
                    1].strip()
            elif '症状:' in line:
                current_prediction['symptoms'] = line.split(':')[1].strip()
            else:
                current_prediction['features'] = line.split(':')[1].strip()

    if current_prediction:
        predictions.append(current_prediction)

    logger.debug(f"Extracted predictions: {predictions}")
    return predictions


def normalize_label(label: str) -> tuple:
    """ラベルを正規化して植物名と状態に分割する"""
    parts = label.split('___')
    plant = parts[0].replace('_', ' ').strip()
    disease = parts[1].replace('_', ' ').strip() if len(
        parts) > 1 else "healthy"

    # 特殊なケースの処理
    if "including sour" in plant.lower():
        plant = "Cherry"
    if "bell" in plant.lower():
        plant = "Pepper"

    return plant, disease


def compare_labels(pred_label: str, true_label: str, label_type: str) -> tuple[bool, float]:
    """ラベルを比較し、一致度も返す"""
    pred = pred_label.lower().strip()
    true = true_label.lower().strip()

    # 完全一致の場合
    if pred == true:
        return True, 1.0

    # 植物名の比較の場合
    if label_type == 'plant':
        # 部分一致を許容
        if pred in true or true in pred:
            return True, 0.8
        return False, 0.0

    # 病気名の比較の場合
    else:
        # 健康状態の比較
        if 'healthy' in pred and 'healthy' in true:
            return True, 1.0

        # 病名の部分一致を確認
        pred_words = set(pred.split())
        true_words = set(true.split())
        common_words = pred_words & true_words

        if common_words:
            # 一致する単語数に基づいて類似度を計算
            similarity = len(common_words) / \
                max(len(pred_words), len(true_words))
            return True, similarity

        return False, 0.0


def process_single_image(args):
    """1枚の画像を処理する"""
    image_path, true_label, labels = args
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            image = load_image(image_path)
            if image is None:
                return None

            # Geminiモデルによる分析
            response = model.generate_content([create_prompt(), image])
            response_text = response.text

            # 予測の抽出
            predictions = extract_predictions(response_text)

            if not predictions:
                raise ValueError("No predictions extracted from response")

            # 真のラベルを植物名と病気名に分割
            true_plant, true_disease = normalize_label(true_label)

            return {
                'image_path': str(image_path),
                'true_label': true_label,
                'true_plant': true_plant,
                'true_disease': true_disease,
                'predictions': predictions,
                'raw_response': response_text
            }

        except Exception as e:
            retry_count += 1
            logger.error(
                f"Error processing {image_path} (attempt {retry_count}): {str(e)}")
            if retry_count < max_retries:
                sleep(WAIT_TIME * (2 ** retry_count))  # 指数バックオフ
            else:
                return None

    return None


def calculate_accuracy_metrics(results: list) -> dict:
    """植物種と病気の両方についてTop-1とTop-5の精度を計算する"""
    total = len(results)
    if total == 0:
        return {
            'plant': {'top1': 0.0, 'top5': 0.0},
            'disease': {'top1': 0.0, 'top5': 0.0}
        }

    plant_correct_top1 = 0
    plant_correct_top5 = 0
    disease_correct_top1 = 0
    disease_correct_top5 = 0

    plant_similarity_sum = 0.0
    disease_similarity_sum = 0.0

    for result in results:
        true_plant, true_disease = normalize_label(result['true_label'])
        predictions = result['predictions']

        logger.debug(f"\nProcessing result for {result['image_path']}")
        logger.debug(
            f"True labels - Plant: {true_plant}, Disease: {true_disease}")

        # Top-1の確認
        if predictions:
            plant_match, plant_similarity = compare_labels(
                predictions[0]['plant'], true_plant, 'plant')
            disease_match, disease_similarity = compare_labels(
                predictions[0]['disease'], true_disease, 'disease')

            if plant_match:
                plant_correct_top1 += 1
                plant_similarity_sum += plant_similarity
                logger.debug(
                    f"Top-1 plant match: {predictions[0]['plant']} == {true_plant} (similarity: {plant_similarity:.2f})")

            if disease_match:
                disease_correct_top1 += 1
                disease_similarity_sum += disease_similarity
                logger.debug(
                    f"Top-1 disease match: {predictions[0]['disease']} == {true_disease} (similarity: {disease_similarity:.2f})")

        # Top-5の確認（確信度による重み付け）
        plant_top5_matches = []
        disease_top5_matches = []

        for pred in predictions[:5]:
            plant_match, plant_sim = compare_labels(
                pred['plant'], true_plant, 'plant')
            disease_match, disease_sim = compare_labels(
                pred['disease'], true_disease, 'disease')

            if plant_match:
                plant_top5_matches.append(
                    (plant_sim, pred.get('confidence', 0) / 100))
            if disease_match:
                disease_top5_matches.append(
                    (disease_sim, pred.get('confidence', 0) / 100))

        if plant_top5_matches:
            plant_correct_top5 += 1
            # 最も高い類似度と確信度の組み合わせを使用
            best_plant_match = max(
                plant_top5_matches, key=lambda x: x[0] * x[1])
            plant_similarity_sum += best_plant_match[0]
            logger.debug("Plant found in Top-5")

        if disease_top5_matches:
            disease_correct_top5 += 1
            best_disease_match = max(
                disease_top5_matches, key=lambda x: x[0] * x[1])
            disease_similarity_sum += best_disease_match[0]
            logger.debug("Disease found in Top-5")

    # 平均類似度を考慮した精度の計算
    metrics = {
        'plant': {
            'top1': (plant_correct_top1 / total) * 100,
            'top5': (plant_correct_top5 / total) * 100,
            'avg_similarity': (plant_similarity_sum / total) * 100
        },
        'disease': {
            'top1': (disease_correct_top1 / total) * 100,
            'top5': (disease_correct_top5 / total) * 100,
            'avg_similarity': (disease_similarity_sum / total) * 100
        }
    }

    logger.debug(f"Final metrics: {metrics}")
    return metrics


def process_dataset() -> dict:
    """データセット全体を処理する"""
    logger.info(f"Processing dataset at: {BASE_DATASET_PATH}")

    # ラベルリストの取得
    labels = get_dataset_labels(BASE_DATASET_PATH)
    logger.info(f"Found {len(labels)} labels in dataset")

    # 画像パスとラベルのリストを作成
    image_data = []
    for label in labels:
        dir_path = Path(BASE_DATASET_PATH) / label
        if not dir_path.exists():
            logger.error(f"Directory does not exist: {dir_path}")
            continue

        for img_name in safe_listdir(dir_path):
            image_path = dir_path / img_name
            if image_path.is_file():
                image_data.append((image_path, label, labels))

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
            sleep(WAIT_TIME)  # API制限対応

    # 精度の計算
    accuracy_metrics = calculate_accuracy_metrics(results)

    logger.info(f"""
    Classification Results:
    Plant Species:
        Top-1 Accuracy: {accuracy_metrics['plant']['top1']:.2f}%
        Top-5 Accuracy: {accuracy_metrics['plant']['top5']:.2f}%
    Disease:
        Top-1 Accuracy: {accuracy_metrics['disease']['top1']:.2f}%
        Top-5 Accuracy: {accuracy_metrics['disease']['top5']:.2f}%
    """)

    return {
        'metrics': accuracy_metrics,
        'detailed_results': results,
        'sampling_info': {
            'total_images': len(image_data),
            'sampled_images': len(sampled_data),
            'sampling_ratio': len(sampled_data) / len(image_data)
        }
    }


def save_results(results: dict, output_dir: str):
    """結果をJSONファイルとして保存する"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"classification_results_{timestamp}.json"

    # 結果の整形
    formatted_results = {
        'metrics': results['metrics'],
        'sampling_info': results['sampling_info'],
        'detailed_results': [
            {
                'image_path': r['image_path'],
                'true_label': r['true_label'],
                'predictions': [
                    {
                        'plant': p['plant'],
                        'disease': p['disease'],
                        'confidence': p['confidence'],
                        'features': p.get('features', '')
                    }
                    for p in r['predictions']
                ]
            }
            for r in results['detailed_results']
        ]
    }

    # JSONファイルとして保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_results, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to: {output_path}")


def main():
    """メイン処理"""
    logger.info("Starting zero-shot classification with Gemini 2.0 Flash")

    try:
        # データセット全体を処理
        results = process_dataset()

        # 結果の保存
        save_results(results, OUTPUT_DIR)

        logger.info("Classification completed successfully")

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise


if __name__ == "__main__":
    main()
