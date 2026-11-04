# テスト（PlantLLM）

development-hub のテスト標準（https://github.com/Y-Kanekoo/development-hub/blob/main/docs/testing-standard.md）に従います。段階は static → unit → api → e2e の順で、前の段階が失敗したら止めます。

## 段階とコマンド

| 段階 | コマンド |
| --- | --- |
| setup | `uv venv --python 3.11 .venv`、`uv pip install --python .venv/bin/python -r requirements-web.txt ruff mypy pytest-cov`、frontend/mobile の `npm ci` |
| static | `uv run --no-project --python .venv/bin/python -m ruff check .`、`uv run --no-project --python .venv/bin/python -m mypy . --ignore-missing-imports`、`gitleaks dir . --no-banner --redact --exit-code 1` |
| unit | `uv run --no-project --python .venv/bin/python -m pytest -q tests/test_diagnosis_fallback.py --cov=app_simple --cov-fail-under=38`、frontend の build |
| api | `uv run --no-project --python .venv/bin/python -m pytest -q tests/test_api.py` |
| e2e | `npx playwright test --grep @smoke`（未実装のため skip） |

まとめて実行: `local-ci run .`、またはコミット後に `local-ci run .`（段階の絞り込み: `--stage static,unit`）。

## カバレッジ

- unit 段階の下限: `app_simple.py` の 38%（2026-11-04 の実測値）。引き下げない
- 下限は引き下げない。計測から外したファイルと理由:（なし）

## 現行コードとの差分

- Python 12 ファイルが Ruff の一括 format check に通らない。今回の変更では既存ファイル全体の整形を行わず、format gate は未導入。`AGENTS.md` に `TODO(autonomous)` を記録した。
- frontend/mobile に lint・テスト用 npm script はなく、local-ci の自動検出は install と frontend build のみ。

## スキップの記録

スキップは `.local-ci.yml` の `skip: true` と `skip_reason` に加えて、この表にも書きます。テストコードに `skip` を追加しません。

| 段階 | 理由 | 解除条件 | 記録日 |
| --- | --- | --- | --- |
| e2e | Playwright と `@smoke` のテストが未実装 | 代表シナリオの E2E テスト追加 | 2026-11-04 |

## 弱体化の確認（レビュー時）

- [ ] `acv check --base origin/main` が合格（テスト削除・件数減少・skip追加・閾値引き下げなし）
- [ ] 期待値を緩めていない（完全一致を部分一致や存在確認にしていない）
- [ ] カバレッジの除外・`.local-ci.yml` の skip・段階の削除を追加していない（追加したなら理由をこの文書に記録）
- [ ] 外部通信はモックかローカルのフェイクだけ
