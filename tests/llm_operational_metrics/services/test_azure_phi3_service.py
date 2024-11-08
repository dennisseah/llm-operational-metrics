from datetime import datetime

import pytest
from azure.ai.inference.models import ChatCompletions, CompletionsUsage, SystemMessage
from pytest_mock import MockerFixture

from llm_operational_metrics.services.azure_phi3_service import (
    AzurePhi3Env,
    AzurePhi3Service,
)


@pytest.mark.asyncio
async def test_generate(mocker: MockerFixture):
    mock_client = mocker.AsyncMock()
    mock_client.complete.return_value = ChatCompletions(
        id="fake",
        created=datetime.now(),
        model="fake",
        usage=CompletionsUsage(completion_tokens=10, prompt_tokens=10, total_tokens=20),
        choices=[],
    )

    mocker.patch(
        "llm_operational_metrics.services.azure_phi3_service.ChatCompletionsClient",
        return_value=mock_client,
    )

    service = AzurePhi3Service(
        AzurePhi3Env(
            azure_phi3_endpoint="https://api.openai.com",
            azure_phi3_key="fake",
        )
    )

    result = await service.generate(
        [SystemMessage(content="Hello there")],
    )

    assert result is not None
    assert result.completion is not None
    assert result.operational_metric is not None
    assert result.model_dump() is not None
