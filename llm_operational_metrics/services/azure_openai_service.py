import time
from dataclasses import dataclass

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from lagom.environment import Env
from openai import AsyncAzureOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from llm_operational_metrics.metric_generator.openai_metric_generator import (
    generate as generate_ops_metric,
)
from llm_operational_metrics.protocols.i_azure_openai_service import (
    AzureOpenAIChatCompletion,
    IAzureOpenAIService,
)


class AzureOpenAIEnv(Env):
    azure_openai_endpoint: str
    azure_openai_key: str | None = None
    azure_openai_api_version: str
    azure_openai_deployed_model_name: str


@dataclass
class AzureOpenAIService(IAzureOpenAIService):
    env: AzureOpenAIEnv

    def __post_init__(self):
        if self.env.azure_openai_key is None:
            azure_credential = DefaultAzureCredential()
            token_provider = get_bearer_token_provider(
                azure_credential, "https://cognitiveservices.azure.com/.default"
            )
            self.client = AsyncAzureOpenAI(
                api_version=self.env.azure_openai_api_version,
                azure_endpoint=self.env.azure_openai_endpoint,
                azure_ad_token_provider=token_provider,
            )
        else:
            self.client = AsyncAzureOpenAI(
                api_key=self.env.azure_openai_key,
                api_version=self.env.azure_openai_api_version,
                azure_endpoint=self.env.azure_openai_endpoint,
            )

    async def generate(
        self, prompts: list[ChatCompletionMessageParam], **kwargs
    ) -> AzureOpenAIChatCompletion:
        start_time = time.time()

        result: ChatCompletion = await self.client.chat.completions.create(
            model=self.env.azure_openai_deployed_model_name,
            messages=prompts,
            **kwargs,
        )
        end_time = time.time()

        return AzureOpenAIChatCompletion(
            completion=result,
            operational_metric=generate_ops_metric(result, int(end_time - start_time)),
        )
