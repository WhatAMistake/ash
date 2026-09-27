"""
Core logic for Ash text generation.
Separated from bot logic to be used by both Telegram bot and Web API.
"""

import os
from typing import Optional, List, Dict
from dataclasses import dataclass

try:
    from llm_completion import create_chat_completion, visible_completion_text, log_completion_error
except ImportError:
    from src.llm_completion import create_chat_completion, visible_completion_text, log_completion_error

@dataclass
class TextGenerationResult:
    content: str
    model: str
    truncated: bool = False

class TextGenerator:
    """Generates text using LLM (OpenAI/DeepSeek)."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: str = "deepseek-v3.2",
        language: str = "ru"
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.model = model
        self.language = language
        self.client = None
        self._init_client()
        
    def _init_client(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_base,
                timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
                max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")))
        except ImportError:
            print("openai not installed. pip install openai")
            self.client = None
            
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> TextGenerationResult:
        """Generate text based on prompt."""
        if not self.client:
            return TextGenerationResult(content="Error: LLM client not initialized.", model="none")
            
        clean_instr = "ВАЖНО: Только чистый текст. Без звёздочек, без смайликов, без декоративных элементов, без эмодзи, без markdown-форматирования." if self.language == "ru" else "IMPORTANT: Clean text only. No asterisks, no emojis, no decorative elements, no markdown formatting."
        
        full_system = clean_instr
        if system_instruction:
            full_system = f"{system_instruction}\n\n{clean_instr}"
            
        messages = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = create_chat_completion(
                self.client,
                reasoning_policy=os.getenv("LLM_REASONING_POLICY", "off"),
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = visible_completion_text(response)
            return TextGenerationResult(content=content, model=self.model)
        except Exception as e:
            log_completion_error(e)
            return TextGenerationResult(content="Generation unavailable. Please try again later.", model="error")

    def generate_story_start(self, idea: str) -> str:
        """Generate a short story start based on an idea."""
        prompt = f"Напиши начало истории (до 1000 знаков) на основе этой идеи: {idea}. Стиль: художественный, атмосферный. Без вступлений, сразу к делу."
        if self.language != "ru":
            prompt = f"Write a story start (up to 1000 chars) based on this idea: {idea}. Style: literary, atmospheric. No intro, jump right in."
            
        result = self.generate(prompt, temperature=0.8)
        return result.content
