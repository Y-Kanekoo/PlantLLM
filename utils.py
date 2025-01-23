import os
import json
import csv
import random
from datetime import datetime
from pathlib import Path
import logging
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
import torch

# 環境変数の読み込み
load_dotenv()

# ロギングの設定
logging.basicConfig(
    level=logging.DEBUG,  # INFOからDEBUGに変更
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('classification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# GPUの設定
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    gpu_properties = torch.cuda.get_device_properties(DEVICE)
    logger.info(
        f"Total memory: {gpu_properties.total_memory / 1024**2:.2f} MB")
else:
    logger.info("Using CPU")

# Gemini APIの設定
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))


def init_model(model_name: str = 'models/gemini-pro-vision'):
    """Geminiモデルを初期化する

    Args:
        model_name (str): モデル名

    Returns:
        GenerativeModel: 初期化されたモデル
    """
    logger.info(f"Initializing Gemini model: {model_name}")
    return genai.GenerativeModel(model_name)


# デフォルトモデルの初期化
model = init_model()

# 標準ラベルリスト
STANDARD_LABELS = [
    "Apple - Apple Scab",
    "Apple - Black Rot",
    "Apple - Cedar Apple Rust",
    "Apple - Healthy",
    "Blueberry - Healthy",
    "Cherry (including sour) - Powdery Mildew",
    "Cherry (including sour) - Healthy",
    "Corn (maize) - Cercospora Leaf Spot (Gray Leaf Spot)",
    "Corn (maize) - Common Rust",
    "Corn (maize) - Northern Leaf Blight",
    "Corn (maize) - Healthy",
    "Grape - Black Rot",
    "Grape - Esca (Black Measles)",
    "Grape - Leaf Blight (Isariopsis Leaf Spot)",
    "Grape - Healthy",
    "Orange - Huanglongbing (Citrus Greening)",
    "Peach - Bacterial Spot",
    "Peach - Healthy",
    "Pepper (bell) - Bacterial Spot",
    "Pepper (bell) - Healthy",
    "Potato - Early Blight",
    "Potato - Late Blight",
    "Potato - Healthy",
    "Raspberry - Healthy",
    "Soybean - Healthy",
    "Squash - Powdery Mildew",
    "Strawberry - Leaf Scorch",
    "Strawberry - Healthy",
    "Tomato - Bacterial Spot",
    "Tomato - Early Blight"
]


def normalize_path(path: str) -> Path:
    """パスを正規化する

    Args:
        path (str): 正規化するパス

    Returns:
        Path: 正規化されたPathオブジェクト
    """
    # 文字列をPathオブジェクトに変換
    path_obj = Path(path)
    logger.debug(f"Normalizing path: {path} -> {path_obj}")
    return path_obj


def safe_listdir(path: str) -> list:
    """安全にディレクトリの内容を取得する

    Args:
        path (str): 対象ディレクトリのパス

    Returns:
        list: ディレクトリ内のファイル・ディレクトリ名のリスト
    """
    try:
        path_obj = normalize_path(path)
        if not path_obj.exists():
            logger.error(f"Directory does not exist: {path_obj}")
            return []
        contents = os.listdir(path_obj)
        logger.debug(
            f"Listed directory {path_obj}: {len(contents)} items found")
        return contents
    except Exception as e:
        logger.error(f"Error listing directory {path}: {str(e)}")
        return []


def load_image(image_path: str, max_size: int = 1024) -> Image.Image:
    """画像を読み込み、必要に応じてリサイズする"""
    try:
        path_obj = normalize_path(image_path)
        logger.debug(f"Loading image from: {path_obj}")
        image = Image.open(path_obj)

        # アスペクト比を保持しながらリサイズ
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.debug(f"Resized image from {image.size} to {new_size}")

        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        return None


def get_dataset_labels(dataset_path: str) -> list:
    """データセットからラベルリストを生成する"""
    path_obj = normalize_path(dataset_path)
    logger.debug(f"Getting labels from: {path_obj}")

    labels = []
    for dir_name in safe_listdir(path_obj):
        dir_path = path_obj / dir_name
        if dir_path.is_dir():
            # ラベル名を整形（アンダースコアの削除、読みやすい形式に変更）
            label = dir_name
            labels.append(label)
            logger.debug(f"Found label: {label}")

    sorted_labels = sorted(labels)
    logger.debug(f"Total labels found: {len(sorted_labels)}")
    return sorted_labels


def get_dataset_types() -> list:
    """利用可能なデータセットの種類を取得する"""
    return ['train', 'valid']


def get_dataset_path(base_path: str, dataset_type: str) -> Path:
    """データセットの種類に応じたパスを取得する"""
    path_obj = normalize_path(base_path) / dataset_type
    logger.debug(f"Generated dataset path: {path_obj}")
    return path_obj


def get_all_dataset_labels(base_path: str) -> dict:
    """すべての種類のデータセットからラベルリストを生成する"""
    dataset_labels = {}
    for dataset_type in get_dataset_types():
        dataset_path = get_dataset_path(base_path, dataset_type)
        dataset_labels[dataset_type] = get_dataset_labels(dataset_path)
    return dataset_labels


def save_results(results: dict, output_dir: str, prefix: str):
    """評価結果を保存する"""
    # Windowsのパス区切り文字に対応
    output_dir = str(Path(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # JSON形式で結果を保存
    json_path = str(Path(output_dir) / f'{prefix}_{timestamp}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # CSV形式で詳細結果を保存
    csv_path = str(Path(output_dir) / f'{prefix}_{timestamp}.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=['image_path', 'true_label', 'predicted_label', 'confidence'])
        writer.writeheader()
        for result in results['detailed_results']:
            writer.writerow(result)

    logger.info(f"Results saved to {output_dir}")


def normalize_label(label: str) -> str:
    """ラベルを標準形式に正規化する

    Args:
        label (str): 正規化するラベル

    Returns:
        str: 正規化されたラベル
    """
    # ディレクトリ名からラベルへの変換
    label = label.replace('___', ' - ')
    label = label.replace('_', ' ')

    # 標準ラベルとの完全一致を確認
    if label in STANDARD_LABELS:
        return label

    # 部分一致で最も近いラベルを探す
    for std_label in STANDARD_LABELS:
        if label.lower() in std_label.lower() or std_label.lower() in label.lower():
            return std_label

    logger.warning(f"Unknown label format: {label}")
    return label


def get_top_k_predictions(response_text: str, k: int = 5) -> list:
    """モデルの応答からTop-k予測を抽出する

    Args:
        response_text (str): モデルの応答テキスト
        k (int): 取得する予測数

    Returns:
        list: (ラベル, 確信度)のタプルのリスト
    """
    predictions = []
    current_label = None
    current_confidence = None

    for line in response_text.split('\n'):
        if '選択したラベル:' in line:
            current_label = line.split(':')[1].strip()
        elif '確信度:' in line and current_label:
            try:
                current_confidence = float(
                    line.split(':')[1].strip().replace('%', ''))
                predictions.append((current_label, current_confidence))
                current_label = None
                current_confidence = None
            except ValueError:
                continue

    # 確信度で降順ソート
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:k]


def calculate_metrics(predictions: list, true_labels: list, raw_responses: list = None) -> dict:
    """評価指標を計算する

    Args:
        predictions (list): 予測ラベルのリスト
        true_labels (list): 正解ラベルのリスト
        raw_responses (list, optional): モデルの生の応答テキストのリスト

    Returns:
        dict: 評価指標
    """
    normalized_predictions = [normalize_label(
        p) if p else None for p in predictions]
    normalized_true_labels = [normalize_label(t) for t in true_labels]

    # Top-1精度の計算
    total = len(normalized_predictions)
    correct = sum(1 for p, t in zip(normalized_predictions,
                  normalized_true_labels) if p == t)
    accuracy = correct / total if total > 0 else 0

    # Top-3とTop-5精度の計算
    top3_correct = 0
    top5_correct = 0
    all_predictions = []  # すべての候補を保存

    if raw_responses:
        for true_label, response_text in zip(normalized_true_labels, raw_responses):
            if response_text:
                predictions = get_top_k_predictions(
                    response_text, k=5)  # 上位5件まで取得
                pred_labels = [normalize_label(p[0]) for p in predictions]

                # すべての候補を保存
                all_predictions.append({
                    'true_label': true_label,
                    'predictions': [{'label': p[0], 'confidence': p[1]} for p in predictions]
                })

                # Top-3とTop-5の正解をカウント
                if true_label in pred_labels[:3]:
                    top3_correct += 1
                if true_label in pred_labels:
                    top5_correct += 1

    top3_accuracy = top3_correct / total if total > 0 else 0
    top5_accuracy = top5_correct / total if total > 0 else 0

    # ラベルごとの精度計算
    label_metrics = {label: {'correct': 0, 'total': 0, 'accuracy': 0.0,
                             'top3_correct': 0, 'top3_accuracy': 0.0,
                             'top5_correct': 0, 'top5_accuracy': 0.0}
                     for label in STANDARD_LABELS}

    # 各ラベルのTop-N精度を計算
    if raw_responses:
        for true_label, response_text in zip(normalized_true_labels, raw_responses):
            if response_text and true_label in label_metrics:
                predictions = get_top_k_predictions(response_text)
                pred_labels = [normalize_label(p[0]) for p in predictions]

                if true_label in pred_labels[:3]:
                    label_metrics[true_label]['top3_correct'] += 1
                if true_label in pred_labels:
                    label_metrics[true_label]['top5_correct'] += 1

    # ラベルごとの精度を計算
    for label in STANDARD_LABELS:
        metrics = label_metrics[label]
        if metrics['total'] > 0:
            metrics['accuracy'] = metrics['correct'] / metrics['total']
            metrics['top3_accuracy'] = metrics['top3_correct'] / \
                metrics['total']
            metrics['top5_accuracy'] = metrics['top5_correct'] / \
                metrics['total']

    return {
        'total_samples': total,
        'correct_predictions': correct,
        'accuracy': accuracy,
        'top3_correct': top3_correct,
        'top3_accuracy': top3_accuracy,
        'top5_correct': top5_correct,
        'top5_accuracy': top5_accuracy,
        'label_metrics': label_metrics,
        'all_predictions': all_predictions,  # すべての候補を含む
        'confusion_matrix': {
            'true_labels': normalized_true_labels,
            'predicted_labels': normalized_predictions
        }
    }


def save_results_by_type(results: dict, output_dir: str, prefix: str):
    """結果をJSONとCSVファイルに保存する

    Args:
        results (dict): 保存する結果
        output_dir (str): 出力ディレクトリ
        prefix (str): ファイル名のプレフィックス
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 結果ディレクトリの作成
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 結果の保存
    output_base = f"{prefix}_{timestamp}"

    # JSON形式で保存
    json_path = Path(output_dir) / f"{output_base}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {json_path}")

    # CSVファイルの作成
    csv_path = Path(output_dir) / f"{output_base}.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Image Path', 'True Label', 'Predicted Label'])
        for result in results['all']['detailed_results']:
            writer.writerow([
                result['image_path'],
                result['true_label'],
                result['predicted_label']
            ])
    logger.info(f"CSV results saved to {csv_path}")


def sample_images(image_paths: list, n_samples: int = 1000, seed: int = 2025) -> list:
    """画像パスのリストからランダムにサンプリングする

    Args:
        image_paths (list): 画像パスのリスト
        n_samples (int, optional): サンプリングする画像数. Defaults to 1000.
        seed (int, optional): 乱数シード. Defaults to 2025.

    Returns:
        list: サンプリングされた画像パスのリスト
    """
    random.seed(seed)
    if len(image_paths) <= n_samples:
        logger.warning(
            f"要求されたサンプル数({n_samples})が全画像数({len(image_paths)})より多いため、全画像を使用します。")
        return image_paths

    sampled = random.sample(image_paths, n_samples)
    logger.info(
        f"Sampled {len(sampled)} images from {len(image_paths)} total images")
    return sampled


def extract_plant_name(label: str) -> str:
    """ラベルから植物名を抽出する

    Args:
        label (str): ラベル（例: "Tomato - Early Blight"）

    Returns:
        str: 植物名（例: "Tomato"）
    """
    return label.split(' - ')[0] if ' - ' in label else label


def extract_disease_name(label: str) -> str:
    """ラベルから病気名を抽出する

    Args:
        label (str): ラベル（例: "Tomato - Early Blight"）

    Returns:
        str: 病気名（例: "Early Blight"）または"healthy"
    """
    parts = label.split(' - ')
    return parts[1] if len(parts) > 1 else "healthy"


def calculate_plant_metrics(predicted_labels: list, true_labels: list, raw_responses: list) -> dict:
    """植物種の分類精度を計算する

    Args:
        predicted_labels (list): 予測されたラベルのリスト
        true_labels (list): 正解のラベルのリスト
        raw_responses (list): モデルの生の応答のリスト

    Returns:
        dict: 精度指標を含む辞書
    """
    if not predicted_labels or not true_labels or len(predicted_labels) != len(true_labels):
        logger.error("Invalid input for calculate_plant_metrics")
        return {'accuracy': 0.0, 'top5_accuracy': 0.0}

    correct = 0
    top5_correct = 0
    total = len(predicted_labels)

    for pred, true, response in zip(predicted_labels, true_labels, raw_responses):
        # 植物種のみを抽出して比較
        pred_plant = extract_plant_name(pred) if pred else ""
        true_plant = extract_plant_name(true)

        # 通常の精度計算
        if pred_plant == true_plant:
            correct += 1

        # Top-5精度の計算
        top5_predictions = get_top_k_predictions(response)
        top5_plants = [extract_plant_name(p[0]) for p in top5_predictions]
        if true_plant in top5_plants:
            top5_correct += 1

    accuracy = correct / total if total > 0 else 0.0
    top5_accuracy = top5_correct / total if total > 0 else 0.0

    return {
        'accuracy': accuracy,
        'top5_accuracy': top5_accuracy
    }


def calculate_disease_metrics(predicted_labels: list, true_labels: list, raw_responses: list) -> dict:
    """病気の分類精度を計算する

    Args:
        predicted_labels (list): 予測されたラベルのリスト
        true_labels (list): 正解のラベルのリスト
        raw_responses (list): モデルの生の応答のリスト

    Returns:
        dict: 精度指標を含む辞書
    """
    if not predicted_labels or not true_labels or len(predicted_labels) != len(true_labels):
        logger.error("Invalid input for calculate_disease_metrics")
        return {'accuracy': 0.0, 'top5_accuracy': 0.0}

    correct = 0
    top5_correct = 0
    total = len(predicted_labels)

    for pred, true, response in zip(predicted_labels, true_labels, raw_responses):
        # 病気名のみを抽出して比較
        pred_disease = extract_disease_name(pred) if pred else ""
        true_disease = extract_disease_name(true)

        # 通常の精度計算
        if pred_disease == true_disease:
            correct += 1

        # Top-5精度の計算
        top5_predictions = get_top_k_predictions(response)
        top5_diseases = [extract_disease_name(p[0]) for p in top5_predictions]
        if true_disease in top5_diseases:
            top5_correct += 1

    accuracy = correct / total if total > 0 else 0.0
    top5_accuracy = top5_correct / total if total > 0 else 0.0

    return {
        'accuracy': accuracy,
        'top5_accuracy': top5_accuracy
    }
