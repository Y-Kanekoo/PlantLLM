"""空の診断応答に対するモデル切り替えを確認する。"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from app_simple import run_diagnosis


@pytest.fixture
def model_candidates() -> list[dict[str, object]]:
    return [
        {"config": {"name": "first", "description": "最初", "retry_count": 1}},
        {"config": {"name": "second", "description": "次", "retry_count": 1}},
    ]


@pytest.mark.asyncio
async def test_empty_response_falls_back_to_next_model(
    model_candidates: list[dict[str, object]],
) -> None:
    with (
        patch("app_simple.genai_client", object()),
        patch("app_simple.resolve_models", return_value=model_candidates),
        patch("app_simple.rate_manager.allow", return_value=True),
        patch("app_simple.rate_manager.record_error") as record_error,
        patch("app_simple.process_image_with_retry", new_callable=AsyncMock) as generate,
    ):
        generate.side_effect = [SimpleNamespace(text=None), SimpleNamespace(text="診断結果")]

        result = await run_diagnosis("診断", b"image")

    assert result.text == "診断結果"
    assert result.model_name == "second"
    assert generate.await_count == 2
    record_error.assert_called_once()


@pytest.mark.asyncio
async def test_all_empty_responses_raise_error(
    model_candidates: list[dict[str, object]],
) -> None:
    with (
        patch("app_simple.genai_client", object()),
        patch("app_simple.resolve_models", return_value=model_candidates),
        patch("app_simple.rate_manager.allow", return_value=True),
        patch("app_simple.rate_manager.record_error"),
        patch("app_simple.process_image_with_retry", new_callable=AsyncMock) as generate,
    ):
        generate.side_effect = [SimpleNamespace(text=""), SimpleNamespace(text=None)]

        with pytest.raises(HTTPException) as error:
            await run_diagnosis("診断", b"image")

    assert error.value.status_code == 500
    assert error.value.detail == "診断結果が空でした"
    assert generate.await_count == 2
