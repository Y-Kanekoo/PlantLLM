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
import threading
import time

# 基本設定
BASE_DATASET_PATH = normalize_path("dataset/PlantVillage_color")
OUTPUT_DIR = normalize_path("results/zero_shot")
BATCH_SIZE = 10  # Gemini 1.5 Flash の制限に対応（15 RPM）
WAIT_TIME = 5    # 5秒間隔で制限内に収める
MAX_DAILY_REQUESTS = 1400  # 1日の制限（1,500）に対する安全マージン
SAMPLES_PER_CLASS = 30  # 各クラスから30枚ずつサンプリング

# Gemini 1.5 Flash の設定
MODEL_NAME = "models/gemini-1.5-flash"
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel(MODEL_NAME)

# API制限の監視用カウンター


class APILimitCounter:
    def __init__(self):
        self.daily_requests = 0
        self.minute_requests = 0
        self.last_reset = time.time()
        self.lock = threading.Lock()

    def increment_and_check(self):
        with self.lock:
            current_time = time.time()
            # 1分経過したらリセット
            if current_time - self.last_reset >= 60:
                self.minute_requests = 0
                self.last_reset = current_time

            # より厳密な制限チェック
            if self.daily_requests >= MAX_DAILY_REQUESTS:
                raise Exception("Daily request limit (1,500) reached")
            if self.minute_requests >= 10:  # 余裕を持たせる
                sleep_time = 60 - (current_time - self.last_reset)
                time.sleep(max(0, sleep_time))
                self.minute_requests = 0
                self.last_reset = time.time()

            self.daily_requests += 1
            self.minute_requests += 1
            return True


# グローバルカウンター
api_counter = APILimitCounter()


def create_prompt() -> str:
    """ゼロショット分類用のプロンプトを生成する"""
    return """あなたは植物病理学の専門家として、農作物の葉の画像を分析します。
画像から観察できる特徴に基づいて、植物の種類と健康状態を判断し、必ず5つの候補を提示してください。
各候補は、観察された特徴の異なる解釈に基づいて生成してください。

分析の基本方針：
1. 植物種の同定（形態学的特徴）
   - 葉の基本構造
     * 単葉か複葉か
     * 葉の形状（楕円形、卵形、心形、掌状など）と大きさ
     * 葉縁の特徴（鋸歯、波状、全縁など）
     * 葉脈のパターン（平行脈、網状脈、羽状脈）
     * 葉の厚みや質感（光沢、毛の有無）
   
   - 配置と構造
     * 葉の配置パターン（互生、対生、輪生）
     * 小葉の数と配置（複葉の場合）
     * 葉柄の特徴（長さ、太さ、色）
     * 特徴的な付属物の有無（托葉、葉耳など）

2. 健康状態の評価（病理学的特徴）
   - 正常な状態の特徴
     * 葉色の均一性と鮮やかさ
     * 組織の健全性（しっかりとした質感）
     * 適切な成長状態（大きさ、形の整い）
   
   - 異常所見の観察
     * 変色（黄化、褐変、壊死など）の範囲と程度
     * 病斑の特徴（形状、色、分布、境界の明瞭さ）
     * 組織の変形や損傷（萎縮、膨潤、穿孔）
     * 病原体の痕跡（菌糸、胞子、菌核）
     * 害虫被害の痕跡（食害痕、虫糞）

必須回答形式：
必ず以下の5つの候補を、確信度の高い順に記述してください。
各候補は異なる特徴の解釈に基づくものとし、類似の症状を持つ可能性のある病気も考慮してください。

1番目の候補:
Plant: [植物種を英語で。一般名と学名を併記。例：Tomato (Solanum lycopersicum)]
Condition: [健康状態または病名を英語で。類似症状との区別点も記載]
Confidence: [0-100の整数値]
Analysis:
- Primary Features: [観察された決定的な形態学的特徴を箇条書きで]
- Health Assessment: [観察された健康状態の特徴を箇条書きで]
- Diagnostic Reasoning: [同定と診断の具体的な根拠を箇条書きで]

2番目から5番目の候補:
[同様の形式で記述。特に以下の点に注意：
- 異なる特徴の解釈に基づく代替的な診断
- 類似の症状を持つ別の病気の可能性
- 形態が似ている別の植物種の可能性]

確信度の判定基準：
90-100: 複数の決定的特徴を明確に確認し、学名レベルでの同定が可能
80-89: 主要な特徴は確認できるが、1-2の特徴が不明確
70-79: 主要な特徴の一部が不明確だが、属レベルでの同定は確実
60-69: 複数の解釈が可能な特徴があるが、有力な候補に絞れる
50-59: 特徴の一部のみ確認可能で、複数の候補が同程度に考えられる
30-49: 特徴が不明確または矛盾する所見あり
0-29: 特徴をほとんど確認できない

重要事項：
1. 必ず5つの候補を挙げること
2. 画像から直接観察できる特徴のみに基づいて判断すること
3. 類似の症状を持つ病気も候補として検討すること
4. 形態が似ている別の植物種の可能性も考慮すること
5. 特に重要な特徴は太字（**）で強調すること
6. 回答は必ず英語で記述すること
7. 各候補は異なる特徴の解釈に基づくこと"""


def extract_predictions(response_text: str) -> list:
    """モデルの応答から予測を抽出する"""
    predictions = []
    current_prediction = {}
    in_prediction = False

    logger.debug(f"Raw response text:\n{response_text}")

    for line in response_text.split('\n'):
        line = line.strip()

        # 新しい予測の開始を検出
        if '番目の候補:' in line or 'Plant:' in line:
            if current_prediction and 'plant' in current_prediction:
                predictions.append(current_prediction)
                current_prediction = {}
            in_prediction = True
            if 'Plant:' in line:
                plant_name = line.split(':')[1].strip()
                current_prediction['plant'] = plant_name
            continue

        if not in_prediction:
            continue

        if 'Plant:' in line:
            plant_name = line.split(':')[1].strip()
            current_prediction['plant'] = plant_name
        elif 'Condition:' in line:
            disease = line.split(':')[1].strip()
            current_prediction['disease'] = disease
        elif 'Confidence:' in line:
            try:
                # パーセント記号と空白を除去して数値に変換
                confidence_str = line.split(
                    ':')[1].strip().replace('%', '').strip()
                confidence = float(confidence_str)
                current_prediction['confidence'] = confidence
            except (ValueError, IndexError) as e:
                logger.warning(f"Error parsing confidence: {e}")
                current_prediction['confidence'] = 0.0
        elif 'Primary Features:' in line:
            try:
                current_prediction['features'] = line.split(':')[1].strip()
            except IndexError:
                current_prediction['features'] = ""
        elif 'Health Assessment:' in line:
            try:
                current_prediction['health_assessment'] = line.split(':')[
                    1].strip()
            except IndexError:
                current_prediction['health_assessment'] = ""
        elif 'Diagnostic Reasoning:' in line:
            try:
                current_prediction['diagnostic_reasoning'] = line.split(':')[
                    1].strip()
            except IndexError:
                current_prediction['diagnostic_reasoning'] = ""

    # 最後の予測を追加
    if current_prediction and 'plant' in current_prediction:
        predictions.append(current_prediction)

    # 予測が空の場合のエラーハンドリング
    if not predictions:
        logger.warning("No predictions could be extracted from the response")
        return []

    # 必須フィールドの確認と補完
    for pred in predictions:
        if 'plant' not in pred:
            pred['plant'] = "Unknown"
        if 'disease' not in pred:
            pred['disease'] = "Unknown"
        if 'confidence' not in pred:
            pred['confidence'] = 0.0

    logger.debug(f"Extracted predictions: {predictions}")
    return predictions


def normalize_label(label: str) -> tuple:
    """ラベルを正規化して植物名と状態に分割する"""
    if '_' not in label:
        return label, "Healthy"

    parts = label.split('_', 1)  # 最初の_でのみ分割
    plant = parts[0].strip()
    disease = parts[1].strip() if len(parts) > 1 else "Healthy"

    # 特殊なケースの処理
    plant_mapping = {
        "BellPepper": "Pepper",
        "Bell_Pepper": "Pepper",
        "Bell Pepper": "Pepper"
    }

    # 植物名の正規化
    plant = plant_mapping.get(plant, plant)

    # 病名の正規化
    disease = disease.replace(" ", "")
    if disease.lower() == "healthy":
        disease = "Healthy"

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
        # 特殊なケースの処理
        plant_aliases = {
            "pepper": ["bellpepper", "bell_pepper", "bell pepper", "capsicum"],
            "tomato": ["tomatoes", "solanum lycopersicum"],
            "potato": ["potatoes", "solanum tuberosum"],
            "strawberry": ["strawberries", "fragaria"],
            "apple": ["malus domestica", "malus pumila"],
            "grape": ["vitis vinifera", "vitis"],
            "corn": ["maize", "zea mays"],
            "cherry": ["prunus avium", "prunus"]
        }

        # エイリアスチェック
        pred_normalized = pred.replace(
            " ", "").replace("_", "").replace("-", "")
        true_normalized = true.replace(
            " ", "").replace("_", "").replace("-", "")

        for base_name, aliases in plant_aliases.items():
            aliases_normalized = [a.replace(" ", "").replace(
                "_", "").replace("-", "") for a in aliases]
            if pred_normalized in aliases_normalized and true_normalized == base_name.replace(" ", ""):
                return True, 0.9
            if true_normalized in aliases_normalized and pred_normalized == base_name.replace(" ", ""):
                return True, 0.9

        # 学名を含む場合の処理
        if "(" in pred and ")" in pred:
            pred_common = pred.split("(")[0].strip()
            return compare_labels(pred_common, true, 'plant')

        # 部分一致チェック（より厳密な条件）
        if (pred in true or true in pred) and len(min(pred, true, key=len)) > 3:
            return True, 0.8

        return False, 0.0

    # 病気名の比較の場合
    else:
        # 健康状態の比較
        healthy_terms = ['healthy', 'normal', 'no disease',
                         'appears healthy', 'good condition']
        if any(term in pred for term in healthy_terms) and 'healthy' in true:
            return True, 1.0

        # 病名の正規化
        pred_words = set(pred.lower().replace(
            "_", " ").replace("-", " ").split())
        true_words = set(true.lower().replace(
            "_", " ").replace("-", " ").split())

        # 特殊なケースの処理
        disease_aliases = {
            "bacterial spot": ["bacterialspot", "bacterial infection", "bacterial disease"],
            "black rot": ["blackrot", "fungal rot", "black fungal infection"],
            "cedar rust": ["cedarrust", "cedar apple rust", "gymnosporangium"],
            "leaf mold": ["leafmold", "fungal mold", "leaf fungus"],
            "leaf spot": ["leafspot", "foliar spot", "leaf lesion"],
            "powdery mildew": ["powderymildew", "white mildew", "powdery fungus"],
            "early blight": ["earlyblight", "alternaria", "early leaf blight"],
            "late blight": ["lateblight", "phytophthora", "late leaf blight"],
            "mosaic virus": ["mosaic", "viral infection", "viral mosaic"],
            "septoria": ["septoria leaf spot", "septoria blight"]
        }

        # エイリアスチェック
        for base_name, aliases in disease_aliases.items():
            base_words = set(base_name.split())
            if pred_words == base_words or true_words == base_words:
                pred_words = base_words
                true_words = base_words
            for alias in aliases:
                alias_words = set(alias.split())
                if pred_words == alias_words and true_words == base_words:
                    return True, 0.9
                if true_words == alias_words and pred_words == base_words:
                    return True, 0.9

        # 共通単語による類似度計算
        common_words = pred_words & true_words
        if common_words:
            # 重要な単語の重み付け
            important_words = {"bacterial", "fungal", "viral",
                               "rust", "blight", "spot", "mold", "mosaic"}
            weighted_common = sum(
                2 if word in important_words else 1 for word in common_words)
            weighted_total = sum(
                2 if word in important_words else 1 for word in pred_words | true_words)
            similarity = weighted_common / weighted_total
            return True, similarity

        return False, 0.0


def process_single_image(args):
    """1枚の画像を処理する"""
    image_path, true_label, labels = args
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            logger.debug(f"Processing image: {image_path}")
            start_time = time.time()

            # API制限のチェック
            if not api_counter.increment_and_check():
                logger.warning("API limit check failed, retrying...")
                retry_count += 1
                continue

            image = load_image(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None

            # Geminiモデルによる分類
            response = model.generate_content(
                [create_prompt(), image],
                generation_config={
                    "temperature": 0.3,  # より決定論的な応答に
                    "top_p": 0.8,
                    "top_k": 40,
                }
            )

            if not response or not response.text:
                logger.warning(f"Empty response for {image_path}")
                retry_count += 1
                continue

            response_text = response.text
            logger.debug(f"Model response: {response_text}")

            # 予測の抽出
            predictions = extract_predictions(response_text)

            if not predictions:
                logger.warning(f"No valid predictions for {image_path}")
                retry_count += 1
                continue

            process_time = time.time() - start_time

            result = {
                'image_path': str(image_path),
                'true_label': true_label,
                'predictions': predictions,
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

            if "429" in error_msg or "rate limit" in error_msg.lower():  # Rate limit error
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
                    (plant_sim, pred.get('confidence', 0) / 100)
                )
            if disease_match:
                disease_top5_matches.append(
                    (disease_sim, pred.get('confidence', 0) / 100)
                )

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

    # 画像パスとラベルのリストを作成（クラスごとに分類）
    class_images = {label: [] for label in labels}
    for label in labels:
        dir_path = Path(BASE_DATASET_PATH) / label
        if not dir_path.exists():
            logger.error(f"Directory does not exist: {dir_path}")
            continue

        for img_name in safe_listdir(dir_path):
            image_path = dir_path / img_name
            if image_path.is_file():
                class_images[label].append((image_path, label, labels))

    # 各クラスから指定枚数をサンプリング
    sampled_data = []
    for label, images in class_images.items():
        if len(images) >= SAMPLES_PER_CLASS:
            sampled = sample_images(
                images, n_samples_per_class=SAMPLES_PER_CLASS)
            sampled_data.extend(sampled)
            logger.info(f"Sampled {len(sampled)} images from class {label}")
        else:
            logger.warning(
                f"Class {label} has fewer than {SAMPLES_PER_CLASS} images")
            sampled_data.extend(images)

    logger.info(f"Total sampled images: {len(sampled_data)}")

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
    Total processed images: {len(results)}
    Plant Species:
        Top-1 Accuracy: {accuracy_metrics['plant']['top1']:.2f}%
        Top-5 Accuracy: {accuracy_metrics['plant']['top5']:.2f}%
        Average Similarity: {accuracy_metrics['plant']['avg_similarity']:.2f}%
    Disease:
        Top-1 Accuracy: {accuracy_metrics['disease']['top1']:.2f}%
        Top-5 Accuracy: {accuracy_metrics['disease']['top5']:.2f}%
        Average Similarity: {accuracy_metrics['disease']['avg_similarity']:.2f}%
    """)

    # 結果の保存
    save_results_by_type(results, OUTPUT_DIR)

    return {
        'results': results,
        'metrics': accuracy_metrics
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


def save_results_by_type(results: list, output_dir: str, prefix: str = "") -> None:
    """結果を種類別に保存する

    Args:
        results: 分類結果のリスト
        output_dir: 出力ディレクトリのパス
        prefix: ファイル名のプレフィックス（デフォルトは空文字）
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{prefix}_results_{timestamp}" if prefix else f"results_{timestamp}"

    # 出力ディレクトリの作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 結果の分類（植物種別、病気別）
    plant_results = {}
    disease_results = {}

    for result in results:
        if not result:
            continue

        true_plant, true_disease = normalize_label(result['true_label'])

        # 植物種別の結果を集計
        if true_plant not in plant_results:
            plant_results[true_plant] = []
        plant_results[true_plant].append(result)

        # 病気別の結果を集計
        if true_disease not in disease_results:
            disease_results[true_disease] = []
        disease_results[true_disease].append(result)

    # 結果の保存
    results_data = {
        'timestamp': timestamp,
        'total_processed': len(results),
        'plant_results': plant_results,
        'disease_results': disease_results,
        'raw_results': results
    }

    output_file = output_path / f"{base_filename}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to: {output_file}")

    # サマリーの作成と保存
    summary = {
        'timestamp': timestamp,
        'total_images': len(results),
        'plant_distribution': {plant: len(items) for plant, items in plant_results.items()},
        'disease_distribution': {disease: len(items) for disease, items in disease_results.items()}
    }

    summary_file = output_path / f"{base_filename}_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"Summary saved to: {summary_file}")


def main():
    """メイン処理"""
    logger.info("Starting zero-shot classification with Gemini 1.5 Flash")

    try:
        process_dataset()
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        raise


if __name__ == "__main__":
    main()
