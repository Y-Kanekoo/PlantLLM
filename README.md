# 植物種分類システム（Plant Species Classification System）

## 概要

このプロジェクトは、Google Gemini Vision Pro を使用して、植物の種類を高精度に分類するシステムです。
画像認識と大規模言語モデルを組み合わせることで、詳細な特徴分析と分類を実現します。

## 主な機能

- 植物種の自動分類
- 特徴ベースの分析
- 信頼度スコアの提供
- バッチ処理による効率的な分類
- 結果の可視化と分析
- 詳細なログ記録

## 技術スタック

- **AI/ML**: Google Gemini Vision Pro (1.5 Flash / 2.0 Flash)
- **画像処理**: Pillow
- **データ分析**: Pandas, NumPy
- **可視化**: Matplotlib, Seaborn
- **進捗管理**: tqdm

# PlantLLM - 植物病気診断システム

Gemini API を使用した植物の病気診断システムです。画像をアップロードすることで、植物の種類と健康状態を診断します。

## 特徴

- 複数の Gemini モデルに対応（2.0 Flash, 1.5 Flash, 1.5 Flash-8B, 1.5 Pro）
- 画像のドラッグ＆ドロップまたはカメラでの撮影に対応
- 診断履歴の保存と管理
- レスポンシブデザイン
- エラー時の自動リトライとフォールバック

## セットアップ手順

### 1. リポジトリのクローン

```bash
git clone https://github.com/[your-username]/plant-classification.git
cd plant-classification
```

### 2. 環境構築

```bash
# 仮想環境の作成と有効化
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# 依存パッケージのインストール
pip install -r requirements.txt
```

### 3. 環境変数の設定

`.env`ファイルを作成し、以下の内容を設定：

```env
GEMINI_API_KEY=your_gemini_api_key
```

### 4. データセットの準備

```bash
python download_dataset.py
```

### 5. 分類の実行

```bash
# 植物種の分類
python plant_species_classification.py

# 結果の可視化
python visualize_results.py
```

## システム構成

### メインコンポーネント

1. **植物種分類（plant_species_classification.py）**

   - 画像の読み込みと前処理
   - Gemini API による特徴抽出
   - バッチ処理による効率的な分類
   - 結果の保存とログ記録

2. **結果可視化（visualize_results.py）**

   - 分類結果の可視化
   - 信頼度分布の分析
   - 処理時間の統計
   - 特徴分析のグラフ化

3. **ユーティリティ（utils.py）**
   - 共通機能の提供
   - エラーハンドリング
   - ヘルパー関数

### ディレクトリ構造

```
.
├── plant_species_classification.py  # メイン分類スクリプト
├── visualize_results.py            # 結果可視化スクリプト
├── utils.py                        # ユーティリティ関数
├── requirements.txt                # 依存パッケージ
├── .env                           # 環境変数
├── dataset/                       # データセット
│   └── PlantVillage_3Variants/
│       ├── color/                # カラー画像
│       ├── grayscale/           # グレースケール画像
│       └── segmented/           # セグメント化画像
├── results/                      # 分類結果
│   ├── analysis_YYYYMMDD_HHMMSS/  # 実行時刻ごとの結果
│   └── visualization/           # 可視化結果
└── logs/                        # ログファイル
```

## API 制限

### Gemini 1.5 Flash

- 15 リクエスト/分（RPM）
- 100 万トークン/分（TPM）
- 1,500 リクエスト/日（RPD）

### Gemini 2.0 Flash

- 10 RPM
- 400 万 TPM
- 1,500 RPD

## 結果フォーマット

分類結果は以下の JSON 形式で保存されます：

```json
{
    "plant_name": {
        "common": "植物の一般名",
        "scientific": "学名",
        "alternatives": ["代替候補1", "代替候補2"]
    },
    "confidence": 0-100の信頼度スコア,
    "features": {
        "leaf_type": "葉の特徴",
        "venation": "葉脈パターン",
        "margin": "葉の縁の特徴"
    },
    "process_time": 処理時間（秒）
}
```

## 制限事項

- 対応画像形式: JPEG、PNG
- 推奨画像サイズ: 1024px 以上
- API 制限に基づく処理制限

## 開発者向け情報

### デバッグ

- 詳細なログ: `classification.log`
- バッチサイズの調整: `batch_size`パラメータ
- 待機時間の調整: `wait_time`パラメータ

### パフォーマンス最適化

- バッチ処理による効率化
- エラーリトライ機能
- 並列処理の実装

## 新機能: 病気診断システム

### インストール

1. 依存パッケージのインストール:

```bash
pip install -r requirements.txt
```

2. サーバーの起動:

```bash
python -m uvicorn app_simple:app --host 0.0.0.0 --port 8000 --reload
```

### 主な機能

- 複数画像の一括診断
- 診断履歴の保存と表示
- 結果の保存と共有
- レスポンシブな UI
- エラー時の自動リカバリー

## ライセンス

MIT ライセンス

## 貢献

バグ報告や機能改善の提案は、Issue やプルリクエストでお願いします。
