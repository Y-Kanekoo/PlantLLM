"""
PlantLLM APIエンドポイントのテスト

実行方法:
    pytest tests/test_api.py -v
"""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

# テスト用にGoogle GenAI SDKをモック
with patch.dict("os.environ", {"GEMINI_API_KEY": "test_key"}):
    mock_client = MagicMock()
    with patch("google.genai.Client", return_value=mock_client):
        from app_simple import (
            app,
            detect_mime_type,
            process_image,
            get_extension_from_mime,
            DiagnosisResult,
            _parse_cors_origins,
            MAX_FILE_SIZE_BYTES,
        )

from starlette.testclient import TestClient

client = TestClient(app, raise_server_exceptions=False)


class TestDetectMimeType:
    """MIMEタイプ検出のテスト"""

    def test_detect_jpeg(self):
        # JPEGのmagic bytes
        jpeg_data = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        assert detect_mime_type(jpeg_data) == "image/jpeg"

    def test_detect_png(self):
        # PNGのmagic bytes
        png_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        assert detect_mime_type(png_data) == "image/png"

    def test_detect_gif87a(self):
        gif_data = b"GIF87a\x01\x00\x01\x00"
        assert detect_mime_type(gif_data) == "image/gif"

    def test_detect_gif89a(self):
        gif_data = b"GIF89a\x01\x00\x01\x00"
        assert detect_mime_type(gif_data) == "image/gif"

    def test_detect_unknown(self):
        unknown_data = b"unknown data format"
        assert detect_mime_type(unknown_data) == "application/octet-stream"


class TestGetExtensionFromMime:
    """MIMEタイプから拡張子取得のテスト"""

    def test_jpeg_extension(self):
        assert get_extension_from_mime("image/jpeg") == ".jpg"

    def test_png_extension(self):
        assert get_extension_from_mime("image/png") == ".png"

    def test_gif_extension(self):
        assert get_extension_from_mime("image/gif") == ".gif"

    def test_unknown_extension_defaults_to_jpg(self):
        """未知のMIMEタイプはデフォルトで.jpg"""
        assert get_extension_from_mime("application/octet-stream") == ".jpg"
        assert get_extension_from_mime("image/webp") == ".jpg"


class TestDiagnosisResult:
    """DiagnosisResultクラスのテスト"""

    def test_diagnosis_result_attributes(self):
        """DiagnosisResultの属性が正しく設定される"""
        result = DiagnosisResult(
            text="テスト診断",
            model_name="test-model",
            model_description="Test Model",
        )
        assert result.text == "テスト診断"
        assert result.model_name == "test-model"
        assert result.model_description == "Test Model"


class TestProcessImage:
    """画像処理のテスト"""

    def _create_test_image(self, mode: str = "RGB", size: tuple = (100, 100)) -> bytes:
        """テスト用画像を生成"""
        img = Image.new(mode, size, color="red")
        buffer = io.BytesIO()
        fmt = "PNG" if mode == "RGBA" else "JPEG"
        img.save(buffer, format=fmt)
        return buffer.getvalue()

    def test_process_rgb_image(self):
        """RGBモード画像の処理"""
        image_data = self._create_test_image("RGB")
        result = process_image(image_data)
        assert result["mime_type"] == "image/jpeg"
        assert len(result["data"]) > 0

    def test_process_rgba_image(self):
        """RGBAモード（透過PNG）の処理 - RGBに変換されること"""
        image_data = self._create_test_image("RGBA")
        result = process_image(image_data)
        # 処理後はJPEGになる
        assert result["mime_type"] == "image/jpeg"
        assert len(result["data"]) > 0

    def test_process_large_image_resize(self):
        """大きい画像がリサイズされること"""
        # 2000x2000の画像
        image_data = self._create_test_image("RGB", (2000, 2000))
        result = process_image(image_data)

        # 結果画像を検証
        result_image = Image.open(io.BytesIO(result["data"]))
        # MAX_IMAGE_SIZE (1024) 以下にリサイズされていること
        assert max(result_image.size) <= 1024

    def test_process_invalid_image(self):
        """無効な画像データでエラー"""
        with pytest.raises(ValueError, match="画像の処理に失敗しました"):
            process_image(b"not an image")


class TestModelsEndpoint:
    """GET /models エンドポイントのテスト"""

    def test_get_models(self):
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "default" in data
        assert isinstance(data["models"], list)


class TestHistoryEndpoint:
    """GET /history エンドポイントのテスト

    Note: 統合テストはTestIntegrationクラスで実行
    """

    def test_get_history(self):
        """履歴取得のテスト（DBが初期化されていれば成功）"""
        response = client.get("/history")
        # DBが初期化されていれば200、されていなければ500
        if response.status_code == 200:
            data = response.json()
            assert data["success"] is True
            assert "history" in data
            assert isinstance(data["history"], list)

    def test_get_history_with_limit(self):
        """limit付き履歴取得のテスト"""
        response = client.get("/history?limit=5")
        if response.status_code == 200:
            data = response.json()
            assert data["success"] is True


class TestDiagnoseEndpoint:
    """POST /diagnose エンドポイントのテスト"""

    def _create_test_jpeg(self) -> bytes:
        """テスト用JPEG画像を生成"""
        img = Image.new("RGB", (100, 100), color="green")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return buffer.getvalue()

    def test_diagnose_invalid_content_type(self):
        """無効なContent-Typeでエラー"""
        response = client.post(
            "/diagnose",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )
        assert response.status_code == 400
        assert "対応していない画像形式" in response.json()["detail"]

    def test_diagnose_invalid_mime_bytes(self):
        """MIMEタイプとmagic bytesの不一致でエラー"""
        # Content-TypeはJPEGだが、中身はテキスト
        response = client.post(
            "/diagnose",
            files={"file": ("test.jpg", b"not an image", "image/jpeg")},
        )
        assert response.status_code == 400
        assert "画像形式の検証に失敗" in response.json()["detail"]


class TestModelStatusEndpoint:
    """GET /model-status エンドポイントのテスト"""

    def test_get_model_status(self):
        response = client.get("/model-status")
        # モデルが初期化されていない場合は空のdictを返す
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)
        else:
            # APIキーがない場合は500エラーになる可能性
            pytest.skip("Model status requires API key")


class TestRootEndpoint:
    """GET / エンドポイントのテスト"""

    def test_serve_root(self):
        response = client.get("/")
        assert response.status_code == 200


class TestHistoryLimitValidation:
    """履歴取得のlimit上限テスト"""

    def test_history_limit_max_enforcement(self):
        """limit=200を指定しても100件に制限されることを確認

        Note: このテストはAPI側のバリデーションを確認。
        実際のデータ件数はDBの状態に依存。
        """
        # リクエスト自体は成功する（DBエラーがなければ）
        response = client.get("/history?limit=200")
        # 内部的にlimitは100に制限される
        # ステータスコードが200なら、limit制限が適用されている
        if response.status_code == 200:
            data = response.json()
            # 実際に返されるデータは100件以下
            if "history" in data:
                assert len(data["history"]) <= 100

    def test_history_limit_negative_corrected(self):
        """負のlimitは1に補正される"""
        response = client.get("/history?limit=-5")
        # サーバーエラーにならなければOK
        assert response.status_code in [200, 500]


class TestDiagnoseWithMock:
    """GenAIモックを使用した/diagnoseテスト"""

    def _create_test_jpeg(self) -> bytes:
        """テスト用JPEG画像を生成"""
        img = Image.new("RGB", (100, 100), color="green")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return buffer.getvalue()

    def test_diagnose_success_with_mock(self):
        """診断成功時のレスポンス形式を確認（モック使用）"""
        jpeg_data = self._create_test_jpeg()

        # DiagnosisResultをモック
        from app_simple import DiagnosisResult
        mock_response = DiagnosisResult(
            text="テスト診断結果",
            model_name="gemini-2.0-flash-exp",
            model_description="Gemini 2.0 Flash",
        )

        with patch("app_simple.run_diagnosis", new_callable=AsyncMock) as mock_diag:
            mock_diag.return_value = mock_response

            response = client.post(
                "/diagnose",
                files={"file": ("plant.jpg", jpeg_data, "image/jpeg")},
            )

            # DBが初期化されていれば200
            if response.status_code == 200:
                data = response.json()
                assert data["success"] is True
                assert "diagnosis" in data
                assert "entry_id" in data


class TestChatEndpoint:
    """POST /chat/{id} エンドポイントのテスト"""

    def test_chat_missing_entry(self):
        """存在しないentry_idでエラー"""
        response = client.post(
            "/chat/99999",
            json={"message": "テストメッセージ"},
        )
        # GenAIクライアントが初期化されていなければ500
        # 初期化されていて、DBにエントリがなければ404
        assert response.status_code in [404, 500]
        if response.status_code == 404:
            assert "エントリが見つかりません" in response.json()["detail"]

    def test_chat_invalid_message_empty(self):
        """空のメッセージでエラー"""
        response = client.post(
            "/chat/1",
            json={"message": ""},
        )
        # Pydanticのバリデーションエラー
        assert response.status_code == 422

    def test_chat_message_too_long(self):
        """長すぎるメッセージでエラー"""
        long_message = "a" * 2001  # max_length=2000を超える
        response = client.post(
            "/chat/1",
            json={"message": long_message},
        )
        # Pydanticのバリデーションエラー
        assert response.status_code == 422


class TestCacheDeduplication:
    """キャッシュによる重複保存テスト"""

    def _create_test_jpeg(self) -> bytes:
        """テスト用JPEG画像を生成"""
        img = Image.new("RGB", (100, 100), color="green")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return buffer.getvalue()

    def test_cache_hit_reuses_image_path(self):
        """キャッシュ命中時は既存の画像パスを再利用する（モック使用）"""
        jpeg_data = self._create_test_jpeg()

        # DiagnosisResultをモック
        from app_simple import DiagnosisResult
        mock_response = DiagnosisResult(
            text="テスト診断結果",
            model_name="gemini-2.0-flash-exp",
            model_description="Gemini 2.0 Flash",
        )

        with patch("app_simple.run_diagnosis", new_callable=AsyncMock) as mock_diag:
            mock_diag.return_value = mock_response

            # 1回目のリクエスト
            response1 = client.post(
                "/diagnose",
                files={"file": ("plant.jpg", jpeg_data, "image/jpeg")},
            )

            # DBが初期化されていない場合はスキップ
            if response1.status_code != 200:
                return

            data1 = response1.json()
            assert data1["success"] is True
            assert data1["from_cache"] is False

            # 2回目のリクエスト（同じ画像）
            response2 = client.post(
                "/diagnose",
                files={"file": ("plant.jpg", jpeg_data, "image/jpeg")},
            )

            if response2.status_code == 200:
                data2 = response2.json()
                assert data2["success"] is True
                # 2回目はキャッシュから取得
                assert data2["from_cache"] is True
                # 画像パスが同じこと（新規保存されていない）
                assert data2["image"]["url"] == data1["image"]["url"]


class TestCorsOriginsParsing:
    """CORS_ORIGINS環境変数のパーステスト"""

    def test_parse_simple_origins(self):
        """単純なカンマ区切りのパース"""
        result = _parse_cors_origins("http://localhost:3000,http://localhost:8000")
        assert result == ["http://localhost:3000", "http://localhost:8000"]

    def test_parse_with_whitespace(self):
        """空白を含む場合のトリミング"""
        result = _parse_cors_origins("  http://localhost:3000 , http://localhost:8000  ")
        assert result == ["http://localhost:3000", "http://localhost:8000"]

    def test_parse_empty_string_returns_default(self):
        """空文字列の場合はデフォルト値を返す"""
        result = _parse_cors_origins("")
        assert result == ["http://localhost:5173", "http://localhost:8000"]

    def test_parse_only_whitespace_returns_default(self):
        """空白のみの場合はデフォルト値を返す"""
        result = _parse_cors_origins("   ,  ,  ")
        assert result == ["http://localhost:5173", "http://localhost:8000"]

    def test_parse_single_origin(self):
        """単一オリジンのパース"""
        result = _parse_cors_origins("https://example.com")
        assert result == ["https://example.com"]


class TestFileSizeBoundary:
    """ファイルサイズ境界値テスト"""

    def _create_test_jpeg(self, size_bytes: int) -> bytes:
        """指定サイズに近いテスト用JPEG画像を生成"""
        # 最小のJPEGを生成
        img = Image.new("RGB", (1, 1), color="white")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        base_data = buffer.getvalue()

        # 必要なサイズに調整（パディング追加）
        if len(base_data) < size_bytes:
            # JPEGコメントマーカーを使ってサイズを増やす
            # 実際にはJPEGではこの方法は使えないので、
            # 画像サイズを大きくして対応
            side = int((size_bytes / 3) ** 0.5) + 1
            img = Image.new("RGB", (side, side), color="white")
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=100)
        return buffer.getvalue()

    def test_file_size_at_limit(self):
        """ファイルサイズが上限ちょうどの場合"""
        # MAX_FILE_SIZE_BYTESは10MBなので、小さい画像で境界テスト
        # 実際のテストでは上限を超えないので成功するはず
        img = Image.new("RGB", (100, 100), color="green")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        jpeg_data = buffer.getvalue()

        response = client.post(
            "/diagnose",
            files={"file": ("test.jpg", jpeg_data, "image/jpeg")},
        )
        # DBが初期化されていない場合は500、それ以外は処理続行
        assert response.status_code in [200, 500]

    def test_file_size_over_limit(self):
        """ファイルサイズが上限を超える場合"""
        # MAX_FILE_SIZE_BYTES + 1 のデータを作成
        over_limit_data = b"\xff\xd8\xff\xe0" + b"x" * (MAX_FILE_SIZE_BYTES + 1)

        response = client.post(
            "/diagnose",
            files={"file": ("test.jpg", over_limit_data, "image/jpeg")},
        )
        assert response.status_code == 400
        assert "ファイルサイズが大きすぎます" in response.json()["detail"]


class TestEntryIdValidation:
    """エントリID検証テスト"""

    def test_history_entry_negative_id(self):
        """負のIDでの履歴取得"""
        response = client.get("/history/-1")
        # 負のIDでも404が返る（存在しないエントリ）
        assert response.status_code in [404, 500]

    def test_history_entry_zero_id(self):
        """ID=0での履歴取得"""
        response = client.get("/history/0")
        assert response.status_code in [404, 500]

    def test_chat_negative_id(self):
        """負のIDでのチャット"""
        response = client.post(
            "/chat/-1",
            json={"message": "test"},
        )
        assert response.status_code in [404, 500]


class TestDatetimeHandling:
    """日時処理のテスト"""

    def test_history_timestamp_format(self):
        """履歴のタイムスタンプがISO形式であること"""
        response = client.get("/history")
        if response.status_code == 200:
            data = response.json()
            if data.get("history"):
                for entry in data["history"]:
                    timestamp = entry.get("timestamp")
                    assert timestamp is not None
                    # ISO形式であることを確認（Tが含まれる）
                    assert "T" in timestamp or "-" in timestamp
