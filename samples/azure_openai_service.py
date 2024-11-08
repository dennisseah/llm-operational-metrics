import json

from llm_operational_metrics.hosting import container
from llm_operational_metrics.protocols.i_azure_openai_service import IAzureOpenAIService


async def main():
    azure_openai_service = container[IAzureOpenAIService]
    result = await azure_openai_service.generate(
        [
            {
                "role": "system",
                "content": "You are a mathemetician. What is the square root of a "
                "negative number.",
            }
        ],
        max_tokens=50,
    )

    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
