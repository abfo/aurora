import json
import logging
from typing import Optional, Any
from openai import OpenAI
from settings import settings
from .base import Tool

sonar_instructions = """You are an artificial intelligence assistant and you answer questions from a user. Your answer
will be read out by voice, so do not include annotations or formatting of any kind in your response. Try to be brief and
answer the question in a single concise paragraph. You never ask clarifying questions, just respond as best as you can."""

class PerplexitySonarSearch(Tool):
    name = "perplexity_sonar_search"

    def is_configured(self) -> bool:
        return bool(settings.perplexity_api_key)

    def manifest(self) -> dict:
        return {
            "name": self.name,
            "type": "function",
            "description": "Answers a question using Internet search via the Perplexity Sonar API. Use this for information after your knowlege cutoff date, to research questions you do not know the answer to, or for local search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user query to search for",
                    },
                },
                "required": ["query"],
            },
        }

    def handle(self, tool_name: str, arguments: Any) -> Optional[str]:
        if tool_name != self.name:
            return None

        arguments = json.loads(arguments)  # Ensure arguments is valid JSON

        query: Optional[str] = arguments.get("query")
        if not query:
            return "Missing required argument: query"

        answer = None
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        sonar_instructions
                    ),
                },
                {   
                    "role": "user",
                    "content": (
                        query
                    ),
                },
            ]
            ppx_client = OpenAI(api_key=settings.perplexity_api_key, base_url="https://api.perplexity.ai")
            response = ppx_client.chat.completions.create(
                model="sonar-pro",
                messages=messages,
            )
            self.analytics.report_event("Sonar")    
            answer = response.choices[0].message.content
        except Exception as err:
            return f'Failed to answer question: {err}'
        return answer

def create_tool(log: Optional[logging.Logger] = None, audio_manager: Any | None = None, **kwargs) -> Tool:
    return PerplexitySonarSearch(log=log, audio_manager=audio_manager, **kwargs)
