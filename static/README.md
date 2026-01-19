# ⚠️ このディレクトリはレガシーです

このディレクトリは旧UIの残骸です。現在は `frontend/` のReactアプリを使用しています。

## 削除について

以下の条件を満たせば安全に削除できます:
- `frontend/dist` をビルド済み
- `app_simple.py` が `frontend/dist` を参照している（現在の実装）

## 削除コマンド

```bash
rm -rf static/ templates/
```
