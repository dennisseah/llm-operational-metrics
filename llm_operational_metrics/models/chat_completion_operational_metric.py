from pydantic import BaseModel


class ChatCompletionOperationalMetric(BaseModel):
    model: str
    time_taken_sec: int
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
