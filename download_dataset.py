import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import logging
from utils import normalize_path
import shutil
from tqdm import tqdm

# ロギングの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# データセットの保存先
DATASET_DIR = normalize_path("dataset")
DATASET_NAME = "vipoooool/new-plant-diseases-dataset"
TARGET_DIR = normalize_path("dataset/PlantVillage_3Variants")


def organize_dataset(source_dir: Path, variant: str):
    """データセットを指定されたバリアント（color/grayscale/segmented）用に整理する"""
    logger.info(f"Organizing {variant} dataset...")

    # 各分割（train、valid、test）を処理
    for split in ['train', 'valid', 'test']:
        split_dir = source_dir / split
        if not split_dir.exists():
            logger.warning(f"Directory not found: {split_dir}")
            continue

        # 各クラスディレクトリを処理
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue

            # ターゲットディレクトリの作成
            target_class_dir = TARGET_DIR / variant / class_dir.name
            target_class_dir.mkdir(parents=True, exist_ok=True)

            # 画像ファイルの移動
            image_files = list(class_dir.glob('*.*'))
            for img_file in tqdm(image_files, desc=f"Processing {class_dir.name} for {variant}"):
                if img_file.is_file():
                    target_path = target_class_dir / img_file.name
                    try:
                        shutil.copy2(str(img_file), str(target_path))
                    except Exception as e:
                        logger.error(f"Error copying {img_file}: {str(e)}")
                        continue

    logger.info(f"{variant} dataset organization completed")


def download_dataset():
    """Kaggle APIを使用してデータセットをダウンロードし、3つのバリアントに整理する"""
    try:
        # 既存のデータセットを削除
        if Path(TARGET_DIR).exists():
            logger.info("Removing existing dataset...")
            shutil.rmtree(str(TARGET_DIR))

        # Kaggle APIの認証
        api = KaggleApi()
        api.authenticate()
        logger.info("Kaggle API authentication successful")

        # 保存先ディレクトリの作成
        Path(DATASET_DIR).mkdir(parents=True, exist_ok=True)

        # データセットのダウンロード
        logger.info(f"Downloading dataset from {DATASET_NAME}...")
        api.dataset_download_files(
            DATASET_NAME,
            path=DATASET_DIR,
            unzip=True
        )

        # ダウンロードしたデータを3つのバリアントに整理
        source_dir = Path(DATASET_DIR) / "New Plant Diseases Dataset(Augmented)" / \
            "New Plant Diseases Dataset(Augmented)"
        if source_dir.exists():
            variants = ['color', 'grayscale', 'segmented']
            for variant in variants:
                organize_dataset(source_dir, variant)
            logger.info(f"Dataset processed and organized in {TARGET_DIR}")
        else:
            raise FileNotFoundError(
                f"Source directory not found: {source_dir}")

        # 元のデータを削除
        cleanup_dir = source_dir.parent
        shutil.rmtree(str(cleanup_dir), ignore_errors=True)
        logger.info("Cleaned up temporary files")

        # 最終確認
        variants = ['color', 'grayscale', 'segmented']
        for variant in variants:
            variant_dir = Path(TARGET_DIR) / variant
            if not any(variant_dir.iterdir()):
                raise Exception(
                    f"No files were created in the {variant} directory")

        logger.info("Dataset download and organization completed successfully")

    except Exception as e:
        logger.error(f"Error in dataset processing: {str(e)}")
        raise


if __name__ == "__main__":
    download_dataset()
