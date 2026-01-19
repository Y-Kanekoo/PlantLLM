#!/bin/bash
# テスト自動実行スクリプト
# Edit/Writeツール使用後にPythonファイルが変更された場合にテストを実行

# 標準入力からツール情報を読み取り
INPUT=$(cat)

# ファイルパスを抽出
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

# ファイルパスが空の場合はスキップ
if [[ -z "$FILE_PATH" ]]; then
    exit 0
fi

# Pythonファイル以外はスキップ
if [[ ! "$FILE_PATH" == *.py ]]; then
    exit 0
fi

# テストファイル自体の変更もテストを実行
cd "$CLAUDE_PROJECT_DIR" || exit 0

# testsディレクトリが存在しない場合はスキップ
if [[ ! -d "tests" ]]; then
    exit 0
fi

# 仮想環境のPythonパスを検出
PYTHON_CMD=""
if [[ -f "venv312/bin/python" ]]; then
    PYTHON_CMD="./venv312/bin/python"
elif [[ -f "venv/bin/python" ]]; then
    PYTHON_CMD="./venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Python not found. Skipping tests."
    exit 0
fi

# pytestが利用可能か確認
if ! $PYTHON_CMD -m pytest --version &> /dev/null; then
    echo "pytest is not installed. Skipping tests."
    exit 0
fi

# テスト実行
echo ""
echo "========================================"
echo "  Auto Test Runner"
echo "  Triggered by: $(basename "$FILE_PATH")"
echo "========================================"
echo ""

# 短い出力形式でテストを実行（失敗時のみ詳細表示）
$PYTHON_CMD -m pytest tests/ -q --tb=short 2>&1 | head -50

# 終了コードに関わらず成功を返す（hookがエラーでブロックされないように）
exit 0
