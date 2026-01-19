"""
pytest設定ファイル

テスト用のフィクスチャとセットアップを定義
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session", autouse=True)
def mock_gemini_api():
    """Gemini APIをモック（全テストセッションで有効）"""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key_for_testing"}):
        yield


@pytest.fixture
def sample_jpeg_bytes():
    """テスト用のJPEG画像バイト列を生成"""
    from io import BytesIO
    from PIL import Image

    img = Image.new("RGB", (100, 100), color="blue")
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def sample_png_rgba_bytes():
    """テスト用の透過PNG画像バイト列を生成"""
    from io import BytesIO
    from PIL import Image

    img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def temp_db_path():
    """一時的なSQLite DBパスを生成"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # テスト後にファイルを削除
    try:
        os.unlink(db_path)
    except OSError:
        pass


@pytest.fixture
def temp_upload_dir():
    """一時的なアップロードディレクトリを生成"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_diagnosis_response():
    """診断結果のモックレスポンスを生成"""
    from app_simple import DiagnosisResult
    return DiagnosisResult(
        text="テスト診断結果\n【植物の種類】トマト\n【健康状態】健康",
        model_name="gemini-2.0-flash-exp",
        model_description="Gemini 2.0 Flash",
    )


@pytest.fixture
def mock_run_diagnosis(mock_diagnosis_response):
    """run_diagnosis関数をモック"""
    async def _mock(*args, **kwargs):
        return mock_diagnosis_response
    return _mock


@pytest.fixture
def mock_genai_client():
    """GenAIクライアントのモック"""
    client = MagicMock()
    client.aio = MagicMock()
    client.aio.models = MagicMock()

    # generate_contentのモック
    mock_response = MagicMock()
    mock_response.text = "テストチャット応答"
    client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    return client
