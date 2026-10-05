"""Shared LLM client, model config, and cost tracking."""
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()

MODEL = "gemini-3.8-flash"

# Paid-tier prices in USD per 1M tokens (Gemini 3.8 Flash, through Dec 31, 2026)
PRICES = {
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75},
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},
}


def cost_usd(input_tokens, output_tokens, thought_tokens, model=MODEL):
    thought_tokens = thought_tokens or 0
    price = PRICES[model]
    input_cost = input_tokens / 1_000_000 * price["input"]
    output_cost = (output_tokens + thought_tokens) / 1_000_000 * price["output"]
    return input_cost + output_cost