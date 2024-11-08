from typing import Protocol

from azure.ai.inference.models import (
    AssistantMessage,
    ChatCompletions,
    SystemMessage,
    UserMessage,
)

from llm_operational_metrics.models.chat_completion_operational_metric import (
    ChatCompletionOperationalMetric,
)


class AzurePhi3ChatCompletion:
    completion: ChatCompletions
    operational_metric: ChatCompletionOperationalMetric

    def __init__(
        self,
        completion: ChatCompletions,
        operational_metric: ChatCompletionOperationalMetric,
    ):
        self.completion = completion
        self.operational_metric = operational_metric

    def model_dump(self):
        return {
            "completion": self.completion.as_dict(),
            "operational_metric": self.operational_metric.model_dump(),
        }


class IAzurePhi3Service(Protocol):
    async def generate(
        self, prompts: list[SystemMessage | AssistantMessage | UserMessage], **kwargs
    ) -> AzurePhi3ChatCompletion: ...
