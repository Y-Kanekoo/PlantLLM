# PlantLLM

PlantLLM は、植物の画像から種類と健康状態を診断する Web アプリです。FastAPI の API と React の UI を分離し、診断履歴とチャットを SQLite に保存します。

## 主な機能

- Gemini 2.0 Flash を優先した診断（必要に応じてフォールバック）
- 画像のドラッグ&ドロップ / 複数枚まとめて診断
- 診断履歴の保存、詳細閲覧、チャット追質問
- React ベースのレスポンシブ UI

## 技術スタック

- Backend: FastAPI + SQLAlchemy + SQLite
- Frontend: React + Vite
- AI: Google Gemini Vision (Flash)

## セットアップ

### 1. 環境変数

`.env` を作成して API キーを設定します。

```env
GEMINI_API_KEY=your_gemini_api_key
```

### 2. Backend の起動

```bash
python -m uvicorn app_simple:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Frontend (開発モード)

```bash
cd frontend
npm install
npm run dev
```

Vite の開発サーバは `http://localhost:5173` で起動します。

### 4. Frontend (本番ビルド)

```bash
cd frontend
npm install
npm run build
```

ビルド後は `frontend/dist` が生成され、FastAPI が静的ファイルとして配信します。

## API エンドポイント

- `POST /diagnose`: 画像診断
- `GET /history`: 診断履歴
- `GET /history/{id}`: 履歴詳細 + チャット履歴
- `POST /chat/{id}`: 追加質問
- `POST /clear-history`: 履歴削除
- `GET /models`: 利用可能なモデル

## Mobile (React Native / Expo)

Web UI で動作確認後、React Native 版を `mobile/` に追加しています。Expo を使って起動します。

```bash
cd mobile
npm install
npm run start
```

モバイル実機でアクセスする場合は、環境変数 `EXPO_PUBLIC_API_BASE_URL` を設定して API の URL を指定してください。

例: macOS のローカル IP を使う場合

```bash
EXPO_PUBLIC_API_BASE_URL=http://192.168.0.10:8000 npm run start
```

## 研究・評価用スクリプト

PlantVillage データセットを使った評価や分析用スクリプトが残っています。アプリ利用だけであれば実行不要です。

- `download_dataset.py`
- `plant_species_classification.py`
- `zero_shot_classification.py`
- `closed_set_classification.py`

## メモ

- 画像は `uploads/` に保存されます。
- SQLite の DB は `plant_diagnosis.db` に保存されます。`DATABASE_URL` で変更可能です。
