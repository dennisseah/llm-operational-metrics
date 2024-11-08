from typing import Protocol

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam,
)
from pydantic import BaseModel

from llm_operational_metrics.models.chat_completion_operational_metric import (
    ChatCompletionOperationalMetric,
)


class AzureOpenAIChatCompletion(BaseModel):
    completion: ChatCompletion
    operational_metric: ChatCompletionOperationalMetric


class IAzureOpenAIService(Protocol):
    async def generate(
        self, prompts: list[ChatCompletionMessageParam], **kwargs
    ) -> AzureOpenAIChatCompletion: ...
