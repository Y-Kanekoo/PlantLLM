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


def create_prompt(labels: list) -> str:
    """分類プロンプトを生成する（ラベルを既知として扱う）

    Args:
        labels (list): 利用可能なラベルのリスト

    Returns:
        str: 生成されたプロンプト
    """
    prompt = f"""この植物の画像を分析し、以下の手順で正確な診断を行ってください。

1. 植物種の特定（最重要）:
   A. 葉の形態学的特徴
      - 葉の配置: 単葉/複葉、対生/互生
      - 葉身の形状: 楕円形/心形/針形など
      - 葉縁の特徴: 鋸歯/全縁/波状など
      - 葉脈のパターン: 平行脈/網状脈/羽状脈
   
   B. 葉の質感と表面特性
      - 表面の光沢: 艶あり/艶なし
      - 毛の有無と特徴
      - 葉の厚さや硬さの印象
   
   C. 植物全体の特徴（可能な場合）
      - 成長パターン
      - 茎の特徴
      - 全体的な色調

2. 健康状態の評価:
   A. 正常な特徴
      - 適切な葉色（種類による）
      - 正常な葉の形状
      - 健康的な生育状態
   
   B. 異常が見られる場合
      - 変色: 黄化/褐変/斑点
      - 変形: 萎縮/巻き/歪み
      - 損傷: 穴/裂け目/壊死

3. 病気の症状分析（該当する場合）:
   A. 病変の特徴
      - 形状: 円形/不規則/同心円状
      - 色: 褐色/黒色/白色など
      - テクスチャ: 粉状/水浸状/隆起
   
   B. 病変の分布
      - 発生位置: 葉縁/葉脈/葉全体
      - 広がりパターン: 散在/集中/進行性
   
   C. 進行段階
      - 初期/中期/後期の判断
      - 重症度の評価

4. 以下のラベルから最も適切なものを選択:
{', '.join(labels)}

回答形式:
1番目の候補:
- 選択したラベル: [ラベル名]
- 確信度: [0-100]
- 主要な識別特徴:
  * 植物種の決定的特徴: [最も重要な形態学的特徴]
  * 健康状態/病気の特徴: [観察された症状や状態]
  * 類似種との区別点: [他の候補との明確な違い]
- 選択理由: [特徴と診断の関連性を具体的に]

[2-5番目も同様の形式]

重要な判断基準:
1. 確信度の判断基準
   - 90-100%: すべての特徴が完全に一致
   - 70-89%: 主要な特徴が一致、わずかな不確実性
   - 50-69%: 重要な特徴の一部が一致
   - 30-49%: 一部の特徴のみ一致
   - 0-29%: 特徴の一致が少ない

2. 注意事項
   - 植物種の特定を最優先すること
   - 不確かな特徴は明確にその旨を記載
   - 複数の可能性がある場合は、それぞれの根拠を明示
   - 画質による制限がある場合は具体的に言及

3. 除外すべき状況
   - 画像が極端に不鮮明
   - 重要な特徴が隠れている
   - 決定的な特徴が確認できない"""

    logger.debug(f"Created detailed prompt with {len(labels)} labels")
    return prompt


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
            response = model.generate_content([create_prompt(labels), image])
            response_text = response.text
            logger.debug(f"Model response: {response_text}")

            # 応答から予測ラベルと確信度を抽出
            predicted_label = None
            confidence = 0.0
            features = ""

            for line in response_text.split('\n'):
                if '選択したラベル:' in line:
                    predicted_label = line.split(':')[1].strip()
                elif '確信度:' in line:
                    try:
                        confidence = float(line.split(
                            ':')[1].strip().replace('%', ''))
                    except ValueError:
                        confidence = 0.0
                elif '特徴:' in line:
                    features = line.split(':')[1].strip()

            process_time = time.time() - start_time

            result = {
                'image_path': str(image_path),
                'true_label': true_label,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'features': features,
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

    # 植物種の精度を計算
    plant_metrics = calculate_plant_metrics(
        [extract_plant_name(r['predicted_label'])
            for r in results if r['predicted_label'] is not None],
        [extract_plant_name(r['true_label'])
            for r in results if r['predicted_label'] is not None],
        [r['raw_response'] for r in results if r['predicted_label'] is not None]
    )

    # 病気の精度を計算
    disease_metrics = calculate_disease_metrics(
        [r['predicted_label']
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
