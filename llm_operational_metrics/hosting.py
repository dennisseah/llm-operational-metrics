"""Defines our top level DI container.
Utilizes the Lagom library for dependency injection, see more at:

- https://lagom-di.readthedocs.io/en/latest/
- https://github.com/meadsteve/lagom
"""

import logging

from dotenv import load_dotenv
from lagom import Container, dependency_definition

from llm_operational_metrics.protocols.i_azure_openai_service import IAzureOpenAIService
from llm_operational_metrics.protocols.i_azure_phi3_service import IAzurePhi3Service

load_dotenv(dotenv_path=".env")


container = Container()
"""The top level DI container for our application."""


# Register our dependencies ------------------------------------------------------------


@dependency_definition(container, singleton=True)
def _() -> logging.Logger:
    return logging.getLogger("llm_operational_metrics")


@dependency_definition(container, singleton=True)
def azure_openai_service() -> IAzureOpenAIService:
    from llm_operational_metrics.services.azure_openai_service import AzureOpenAIService

    return container[AzureOpenAIService]


@dependency_definition(container, singleton=True)
def azure_llama_service() -> IAzurePhi3Service:
    from llm_operational_metrics.services.azure_phi3_service import AzurePhi3Service

    return container[AzurePhi3Service]
