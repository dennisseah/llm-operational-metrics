from openai.types.chat.chat_completion import ChatCompletion

from llm_operational_metrics.models.chat_completion_operational_metric import (
    ChatCompletionOperationalMetric,
)


def generate(completion: ChatCompletion, time_taken_sec: int):
    return ChatCompletionOperationalMetric(
        model=completion.model,
        time_taken_sec=time_taken_sec,
        completion_tokens=(
            completion.usage.completion_tokens if completion.usage else 0
        ),
        prompt_tokens=completion.usage.prompt_tokens if completion.usage else 0,
        total_tokens=completion.usage.total_tokens if completion.usage else 0,
    )
