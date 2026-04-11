import ollama
from typing import Generator
from pathlib import Path
from src.config import LLM_MODEL, LLM_PROMPT_PATH

class LLMWrapper:
    def __init__(
        self,
        model: str = LLM_MODEL,
        system_prompt_path: Path = LLM_PROMPT_PATH,
        max_history: int = 20,
    ):
        self.model       = model
        self.max_history = max_history
        self.history: list[dict] = []
        with open(system_prompt_path, "r") as f:
            self.system_prompt = f.read()

    def _build_messages(self, user_input: str, emotion_context: str | None) -> list[dict]:
        system = self.system_prompt
        if emotion_context:
            system += f"\n\n{emotion_context}"

        messages = [{"role": "system", "content": system}]
        messages += self.history[-self.max_history:]
        messages.append({"role": "user", "content": user_input})
        return messages

    def chat(self, user_input: str, emotion_context: str | None = None) -> str:
        messages = self._build_messages(user_input, emotion_context)

        response = ollama.chat(model=self.model, messages=messages)
        reply = response["message"]["content"]

        self.history.append({"role": "user",      "content": user_input})
        self.history.append({"role": "assistant",  "content": reply})

        return reply

    def stream(self, user_input: str, emotion_context: str | None = None) -> Generator[str, None, str]:
        messages = self._build_messages(user_input, emotion_context)

        full_reply = ""
        for chunk in ollama.chat(model=self.model, messages=messages, stream=True):
            token = chunk["message"]["content"]
            full_reply += token
            yield token

        self.history.append({"role": "user",     "content": user_input})
        self.history.append({"role": "assistant", "content": full_reply})

        return full_reply

    def reset(self):
        self.history.clear()

    def update_system_prompt(self, new_prompt: str):
        self.system_prompt = new_prompt