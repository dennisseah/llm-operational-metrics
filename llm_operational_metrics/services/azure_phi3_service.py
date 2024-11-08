import time
from dataclasses import dataclass

from azure.ai.inference.aio import ChatCompletionsClient
from azure.ai.inference.models import (
    AssistantMessage,
    ChatCompletions,
    SystemMessage,
    UserMessage,
)
from azure.core.credentials import AzureKeyCredential
from lagom.environment import Env

from llm_operational_metrics.metric_generator.inference_chat_completion_metric_generator import (  # noqa E501
    generate as generate_ops_metric,
)
from llm_operational_metrics.protocols.i_azure_phi3_service import (
    AzurePhi3ChatCompletion,
    IAzurePhi3Service,
)


class AzurePhi3Env(Env):
    azure_phi3_endpoint: str
    azure_phi3_key: str


@dataclass
class AzurePhi3Service(IAzurePhi3Service):
    env: AzurePhi3Env

    async def generate(
        self, prompts: list[SystemMessage | AssistantMessage | UserMessage], **kwargs
    ) -> AzurePhi3ChatCompletion:
        start_time = time.time()
        client = ChatCompletionsClient(
            endpoint=self.env.azure_phi3_endpoint,
            credential=AzureKeyCredential(self.env.azure_phi3_key),
        )

        try:
            result: ChatCompletions = await client.complete(messages=prompts, **kwargs)  # type: ignore
            end_time = time.time()

            return AzurePhi3ChatCompletion(
                completion=result,
                operational_metric=generate_ops_metric(
                    result, int(end_time - start_time)
                ),
            )
        finally:
            await client.close()
