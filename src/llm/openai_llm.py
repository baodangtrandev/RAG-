import os
from collections.abc import Generator
from typing import Any

from openai import OpenAI

from src.llm.interface import LLMInterface, Message, ReasoningLevel, ToolCall

LLM_MODEL_NAME = os.environ.get("JUDGE_LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct")
CHEAP_LLM_MODEL_NAME = os.environ.get("JUDGE_LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct")


class OpenAILLM(LLMInterface):
    """OpenAI implementation of the LLM interface using the standard Chat Completions API for vLLM compatibility."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        tools: list[dict] | None = None,
        quiet: bool = False,
        reasoning_level: ReasoningLevel = "medium",
    ):
        # We point to localhost:8000 for vLLM
        self.api_key = "empty"
        self.base_url = "http://localhost:8000/v1"
        self.model = model or LLM_MODEL_NAME
        self.tools = tools
        self.quiet = quiet
        self.reasoning_level = reasoning_level

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _build_input(self, messages: list[Message]) -> list[dict[str, Any]]:
        input_items: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                input_items.append({"role": "system", "content": msg.content})
            elif msg.role == "user":
                input_items.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                input_items.append({"role": "assistant", "content": msg.content})
        return input_items

    def generate(self, messages: list[Message]) -> Generator[str | ToolCall, None, None]:
        if not self.quiet:
            print(f"Waiting on LLM ({self.model})...", flush=True)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._build_input(messages),
            "stream": True,
            "temperature": 0.0,
        }

        try:
            stream = self.client.chat.completions.create(**kwargs)
            for chunk in stream:
                if len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content is not None:
                        if not self.quiet:
                            print(delta.content, end="", flush=True)
                        yield delta.content
        except Exception as e:
            print(f"LLM Error: {e}")
            yield ""
