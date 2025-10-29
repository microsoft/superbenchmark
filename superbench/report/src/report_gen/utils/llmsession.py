# TODO: modify the deployment scripts to include below ENV VAR
# COPILOT_LLM_PROVIDER
# OPENAI_API_KEY

"""Language model session class."""

import os
import time
import openai
from ..utils.logger import logger

class LLMSession:
    """A class to interact with the Azure OpenAI model."""

    def __init__(self):
        # Env Var to set the LLM provider, accepted values are 'openai' or 'azure'
        self.provider = os.environ.get("COPILOT_LLM_PROVIDER")
        logger.info(f'COPILOT LLM Endpoint Provider: {self.provider}')
        self.azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.endpoint = os.environ.get("COPILOT_LLM_ENDPOINT")
        self.model_name = os.environ.get("COPILOT_LLM_MODEL")
        self.model_version = os.environ.get("COPILOT_LLM_VERSION")
        if self.provider == "openai":
            self.model = openai.OpenAI(
                base_url=self.endpoint,
                api_key=self.openai_api_key
            )
        elif self.provider == "azure":
            self.model = openai.AzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.azure_api_key,
                api_version=self.model_version
            )
        else:
            logger.error(f'Unsupported LLM provider: {self.provider}')
            raise ValueError(f'Unsupported LLM provider: {self.provider}')

    def chat(self, system_prompt, user_prompt):
        """Chat with the language model."""
        msg = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        max_retries = 5
        backoff = 2  # Initial backoff in seconds

        for attempt in range(max_retries):
            try:
                if self.provider == "azure":
                    response = self.model.chat.completions.create(
                        model=self.model_name,
                        messages=msg,
                        max_completion_tokens=10000
                    )
                    return response.choices[0].message.content
                elif self.provider == "openai":
                    response = self.model.chat.completions.create(
                        model=self.model_name,
                        messages=msg,
                        max_tokens=10000
                    )
                    return response.choices[0].message.content
                else:
                    logger.error(f"Unsupported LLM provider in chat: {self.provider}")
                    break
            except Exception as e:
                if "429" in str(e):
                    logger.warning(f"429 Too Many Requests: Retrying in {backoff} seconds (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(backoff)
                    backoff *= 2  # Exponential backoff
                else:
                    logger.error(f"Unexpected error: {e}")
                    break

        # If retries are exhausted, return a meaningful fallback string
        logger.error("Exceeded maximum retries for chat request.")
        return "The system is currently overloaded. Please try again later."