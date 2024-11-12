import json

from azure.ai.inference.models import SystemMessage

from llm_operational_metrics.hosting import container
from llm_operational_metrics.protocols.i_azure_phi3_service import IAzurePhi3Service


async def main():
    azure_openai_service = container[IAzurePhi3Service]
    result = await azure_openai_service.generate(
        [
            SystemMessage(
                content="You are a mathemetician. What is the square root of a "
                "negative number.",
            ),
        ]
    )

    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
