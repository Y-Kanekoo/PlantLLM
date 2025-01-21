# シンプル植物病害診断システム（Simple Plant Disease Diagnosis System）

## 概要

このプロジェクトは、Google Gemini Vision Pro と FastAPI を使用して、植物の種類と病気の有無を簡単に診断するシステムです。
画像認識と大規模言語モデルを組み合わせることで、迅速な診断を実現します。

## 主な機能

- 植物の種類の識別
- 病気の有無の判定
- 処理時間の計測

## 技術スタック

- **バックエンド**: FastAPI
- **AI/ML**: Google Gemini Vision Pro
- **画像処理**: Pillow

## セットアップ手順

### 1. リポジトリのクローン

```bash
git clone https://github.com/[your-username]/simple-plant-diagnosis.git
cd simple-plant-diagnosis
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
DEBUG=True
UPLOAD_DIR=uploads
HOST=0.0.0.0
PORT=8000
RELOAD=True
LOG_LEVEL=DEBUG
```

### 4. アプリケーションの起動

```bash
uvicorn app_simple:app --reload
```

## API エンドポイント

### 画像診断

- **エンドポイント**: `/diagnose`
- **メソッド**: POST
- **入力**: 画像ファイル（JPEG、PNG、GIF）
- **出力**:
  - 植物の種類
  - 病気の有無
  - 処理時間

## 制限事項

- 対応画像形式: JPEG、PNG、GIF
- 最大ファイルサイズ: 10MB
- 画像サイズ: 自動的に 1024px にリサイズ

## 開発者向け情報

### ディレクトリ構造

```
.
├── app_simple.py        # メインアプリケーション
├── requirements.txt     # 依存パッケージ
├── .env                # 環境変数
├── uploads/           # アップロードされた画像
└── static/           # 静的ファイル
    ├── css/
    └── js/
```

### デバッグ

- ログファイル: `app.log`
- デバッグモード: `.env`の`DEBUG=True`で有効化

## ライセンス

MIT License

## 貢献

バグ報告や機能改善の提案は、Issue やプルリクエストでお願いします。
