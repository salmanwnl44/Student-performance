import os
import requests
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class LLMService:
    def __init__(self, provider="groq"):
        self.provider = provider.lower()
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Fallback to local Ollama if Groq is requested but no key exists
        if self.provider == "groq" and not self.groq_api_key:
            self.provider = "ollama_local"
            
        if self.provider == "groq":
            self.groq_client = Groq(api_key=self.groq_api_key)

    def generate_response(self, text: str, system_prompt: str = "") -> str:
        if self.provider == "groq":
            return self._call_groq(text, system_prompt)
        elif self.provider in ["ollama_local", "ollama_cloud"]:
            return self._call_ollama(text, system_prompt)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def _call_groq(self, text: str, system_prompt: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=messages,
                model="llama-3.1-8b-instant", # Groq model
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error from Groq: {e}"

    def _call_ollama(self, text: str, system_prompt: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})
        
        url = f"{self.ollama_base_url}/api/chat"
        payload = {
            "model": "llama3", # Default ollama model
            "messages": messages,
            "stream": False
        }
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
        except Exception as e:
            return f"Error from Ollama: {e}"
