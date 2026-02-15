"""
LLM Analysis Module
This is the core module that connects the project to local LLMs running through Ollama.
Instead of writing raw API calls every time i need to send a prompt, this module
provides a clean interface that handles all the communication, error handling,
and performance tracking in one place.

The notebook and dashboard both import from this file. When any part of the project
calls analyzer.analyze(prompt), this module:
    1. Sends the prompt to Ollama's REST API on localhost:11434
    2. Waits for the response
    3. Records performance metrics (latency, token count, tokens/sec)
    4. Returns everything in a structured LLMResult object

I built this as a unified interface that supports multiple providers (Ollama, OpenAI,
Anthropic) so the code doesn't need to change if i switch models. For this project
i'm only using Ollama with llama3.1 and mistral - the OpenAI and Anthropic methods
are included as optional extensions but are never called.

Usage:
    from llm_analysis import LLMAnalyzer
    analyzer = LLMAnalyzer(provider="ollama", model="llama3.1")
    result = analyzer.analyze("your prompt here", "system prompt here")
    print(result.response)         # the LLM's text output
    print(result.latency_seconds)  # how long it took
    print(result.completion_tokens) # how many tokens were generated
"""

import requests
import json
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMResult:
    """
    Data class that stores an LLM response along with its performance metrics.

    Every time we call the LLM, we get back one of these objects. It bundles
    the actual text response together with metadata like how long the call took,
    how many tokens were used, and what the cost was (always $0 for local models).

    The notebook uses these objects to build the model comparison table -
    it pulls latency_seconds and completion_tokens from each result to calculate
    tokens/sec for llama3.1 vs mistral.
    """
    response: str                          # the actual text the LLM generated
    model: str                             # which model produced this (e.g. "llama3.1")
    provider: str                          # which provider (e.g. "ollama")
    prompt_tokens: int = 0                 # how many tokens were in the input prompt
    completion_tokens: int = 0             # how many tokens the LLM generated
    latency_seconds: float = 0.0           # wall clock time for the full request
    cost_usd: float = 0.0                  # monetary cost (always 0.0 for local models)
    metadata: dict = field(default_factory=dict)  # extra info like ollama's internal timing


class LLMAnalyzer:
    """
    Unified interface for sending prompts to LLMs and getting structured results back.

    This class abstracts away the differences between providers. Whether i'm calling
    Ollama locally or OpenAI's cloud API, the calling code looks the same:
        result = analyzer.analyze(prompt, system_prompt)

    For this project, only the Ollama provider is used. The OpenAI and Anthropic
    methods exist as optional extensions in case someone wants to do the optional
    cloud API comparison mentioned in the assignment.
    """

    # pricing table for cloud APIs (not used in this project, kept for reference)
    # these are approximate costs per 1000 tokens
    PRICING = {
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    }

    def __init__(self, provider="ollama", model="llama3.1",
                 ollama_url="http://localhost:11434",
                 api_key=None):
        """
        Initialize the analyzer with a specific provider and model.

        For this project, we always use:
            provider="ollama"
            model="llama3.1" or "mistral"
            ollama_url="http://localhost:11434" (default Ollama port)
            api_key=None (not needed for local models)
        """
        self.provider = provider
        self.model = model
        self.ollama_url = ollama_url
        self.api_key = api_key

    def analyze(self, prompt: str, system_prompt: str = "",
                temperature: float = 0.3, max_tokens: int = 2000) -> LLMResult:
        """
        Main method - send a prompt to the LLM and return the result.

        This is what the notebook and dashboard call for every analysis task.
        It routes to the correct provider method, times the call, and returns
        a LLMResult with the response and metrics.

        Args:
            prompt: the actual question/instruction being sent to the LLM
            system_prompt: sets the LLM's role (e.g. "You are a developer psychology expert")
            temperature: controls randomness. 0.3 = more focused and consistent responses.
                         higher values like 0.7 give more creative but less predictable output.
            max_tokens: cap on how long the response can be (2000 tokens ~ 1500 words)

        Returns:
            LLMResult with the response text and all performance metrics
        """
        start_time = time.time()

        # route to the correct provider based on what was configured
        if self.provider == "ollama":
            result = self._call_ollama(prompt, system_prompt, temperature, max_tokens)
        elif self.provider == "openai":
            result = self._call_openai(prompt, system_prompt, temperature, max_tokens)
        elif self.provider == "anthropic":
            result = self._call_anthropic(prompt, system_prompt, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        # record the total wall clock time
        result.latency_seconds = round(time.time() - start_time, 2)
        return result

    def _call_ollama(self, prompt, system_prompt, temperature, max_tokens) -> LLMResult:
        """
        Send a prompt to Ollama's local REST API.

        This is the method that actually does the work for this project.
        Ollama exposes a chat API at http://localhost:11434/api/chat that
        accepts messages in the same format as OpenAI's chat completions.

        The payload includes:
            - model: which model to use (llama3.1 or mistral)
            - messages: list of system + user messages
            - stream: False so we get the complete response in one shot
            - options: temperature and max token settings

        The response includes token counts and timing data from Ollama's
        internal measurements, which we store in the metadata field.

        If Ollama isn't running, this catches the ConnectionError and returns
        a helpful error message instead of crashing.
        """
        payload = {
            "model": self.model,
            "messages": [],
            "stream": False,         # get full response at once, not streamed
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens   # ollama's name for max_tokens
            }
        }

        # the system prompt sets the LLM's persona/role
        # for example "You are a developer psychology expert" makes it
        # analyze commit messages from a behavioral perspective
        if system_prompt:
            payload["messages"].append({"role": "system", "content": system_prompt})
        payload["messages"].append({"role": "user", "content": prompt})

        try:
            # sending the POST request to Ollama
            # timeout is 120 seconds because some tasks take over a minute
            # on CPU-only hardware
            resp = requests.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=120
            )
            resp.raise_for_status()
            data = resp.json()

            # extracting the response text and token counts from Ollama's response
            response_text = data.get("message", {}).get("content", "")
            eval_count = data.get("eval_count", 0)              # tokens generated
            prompt_eval_count = data.get("prompt_eval_count", 0) # tokens in the prompt

            return LLMResult(
                response=response_text,
                model=self.model,
                provider="ollama",
                prompt_tokens=prompt_eval_count,
                completion_tokens=eval_count,
                cost_usd=0.0,  # local inference is free
                metadata={
                    # ollama provides internal timing in nanoseconds
                    "total_duration_ns": data.get("total_duration", 0),
                    "eval_duration_ns": data.get("eval_duration", 0),
                }
            )
        except requests.exceptions.ConnectionError:
            # this happens when Ollama isn't running
            return LLMResult(
                response=f"ERROR: Cannot connect to Ollama at {self.ollama_url}. "
                         "Make sure Ollama is running (run `ollama serve` in a terminal).",
                model=self.model,
                provider="ollama"
            )
        except Exception as e:
            # catch any other unexpected errors
            return LLMResult(
                response=f"ERROR: {str(e)}",
                model=self.model,
                provider="ollama"
            )

    def _call_openai(self, prompt, system_prompt, temperature, max_tokens) -> LLMResult:
        """
        Call OpenAI's cloud API. NOT USED in this project.

        This method exists as an optional extension. The assignment says:
        "You may use OpenAI/Claude API for comparison on a small subset"
        I chose not to use it since the assignment only requires local LLMs.

        If someone wanted to use it, they would create an analyzer like:
            llm = LLMAnalyzer(provider="openai", model="gpt-4o-mini", api_key="sk-...")
        """
        if not self.api_key:
            return LLMResult(
                response="ERROR: OpenAI API key not provided.",
                model=self.model, provider="openai"
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()

            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            # calculating the dollar cost based on token usage
            pricing = self.PRICING.get(self.model, {"input": 0, "output": 0})
            cost = (prompt_tokens / 1000 * pricing["input"] +
                    completion_tokens / 1000 * pricing["output"])

            return LLMResult(
                response=data["choices"][0]["message"]["content"],
                model=self.model,
                provider="openai",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=round(cost, 6)
            )
        except Exception as e:
            return LLMResult(
                response=f"ERROR: {str(e)}",
                model=self.model, provider="openai"
            )

    def _call_anthropic(self, prompt, system_prompt, temperature, max_tokens) -> LLMResult:
        """
        Call Anthropic's Claude API. NOT USED in this project.

        Same as OpenAI above - this is an optional extension that's never called.
        Kept here in case someone wants to compare local models against Claude.
        """
        if not self.api_key:
            return LLMResult(
                response="ERROR: Anthropic API key not provided.",
                model=self.model, provider="anthropic"
            )

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()

            usage = data.get("usage", {})
            return LLMResult(
                response=data["content"][0]["text"],
                model=self.model,
                provider="anthropic",
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
                cost_usd=0.0
            )
        except Exception as e:
            return LLMResult(
                response=f"ERROR: {str(e)}",
                model=self.model, provider="anthropic"
            )

    @staticmethod
    def check_ollama_models(ollama_url="http://localhost:11434"):
        """
        Check which models are currently downloaded in Ollama.

        Calls the /api/tags endpoint which returns a list of all available models.
        The notebook uses this at the start to verify that llama3.1 and mistral
        are pulled and ready to use before running the analysis tasks.

        Returns a list of model names like ['llama3.1:latest', 'mistral:latest']
        """
        try:
            resp = requests.get(f"{ollama_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                return [m["name"] for m in models]
        except Exception:
            pass
        return []


class ModelComparator:
    """
    Utility class for comparing multiple LLM models on the same prompts.

    This is used in the notebook's model comparison section. It takes a list
    of model configurations, creates an LLMAnalyzer for each one, and provides
    methods to run the same prompt through all of them and collect the results
    into a comparison table.

    For this project, it compares llama3.1 vs mistral on tasks like sentiment
    analysis, topic clustering, and career progression.
    """

    def __init__(self, models: list[dict]):
        """
        Set up analyzers for each model to compare.

        Args:
            models: list of dicts, each with provider and model name
                    Example: [
                        {"provider": "ollama", "model": "llama3.1"},
                        {"provider": "ollama", "model": "mistral"},
                    ]
        """
        self.analyzers = {}
        for m in models:
            # creating a unique key like "ollama:llama3.1" for each model
            key = f"{m['provider']}:{m['model']}"
            self.analyzers[key] = LLMAnalyzer(
                provider=m["provider"],
                model=m["model"],
                api_key=m.get("api_key"),
                ollama_url=m.get("ollama_url", "http://localhost:11434")
            )

    def compare(self, prompt: str, system_prompt: str = "",
                temperature: float = 0.3) -> dict[str, LLMResult]:
        """
        Run the exact same prompt through every configured model.

        This ensures a fair comparison since all models receive identical input.
        Returns a dictionary mapping each model's key to its LLMResult.
        """
        results = {}
        for key, analyzer in self.analyzers.items():
            print(f"  Running: {key}...")
            results[key] = analyzer.analyze(prompt, system_prompt, temperature)
            print(f"    -> {results[key].latency_seconds}s, "
                  f"{results[key].completion_tokens} tokens")
        return results

    def comparison_table(self, results: dict[str, LLMResult]) -> list[dict]:
        """
        Convert comparison results into a list of dicts that can be turned
        into a pandas DataFrame for display.

        Calculates tokens/sec from the latency and token count, and includes
        a preview of each model's response (first 200 characters).
        """
        rows = []
        for key, r in results.items():
            tokens_per_sec = (
                r.completion_tokens / r.latency_seconds
                if r.latency_seconds > 0 else 0
            )
            rows.append({
                "Model": key,
                "Provider": r.provider,
                "Latency (s)": r.latency_seconds,
                "Prompt Tokens": r.prompt_tokens,
                "Completion Tokens": r.completion_tokens,
                "Tokens/sec": round(tokens_per_sec, 1),
                "Cost (USD)": r.cost_usd,
                "Response Preview": r.response[:200] + "..." if len(r.response) > 200 else r.response
            })
        return rows