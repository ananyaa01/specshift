import time
from dotenv import load_dotenv
from google import genai

# Load GEMINI_API_KEY from .env into the environment
load_dotenv()

# The client reads GEMINI_API_KEY automatically
client = genai.Client()

MODEL = "gemini-3.1-flash-lite"

# System instruction: who the model is + precise definitions of the labels
SYSTEM = (
    "You are a payments compliance analyst. "
    "You compare two versions of a payment-scheme clause and decide whether "
    "the change is COSMETIC (same obligation, different wording) or "
    "MATERIAL (the obligation, deadline, amount, or requirement changed)."
)

V1 = "The receiving bank shall return a rejected payment within 30 calendar days."
V2 = "The receiving bank must return a rejected payment within 15 calendar days."



INPUT_PRICE_PER_M = 0.75
OUTPUT_PRICE_PER_M = 3.75 

def cost_usd(input_tokens, output_tokens, thought_tokens):
    thought_tokens = thought_tokens or 0                     # None → 0
    input_cost = input_tokens / 1_000_000 * INPUT_PRICE_PER_M
    output_cost = (output_tokens + thought_tokens) / 1_000_000 * OUTPUT_PRICE_PER_M
    return input_cost + output_cost

def classify(v1, v2, temperature=0, thinking="low"):
    prompt = f"Version 1: {v1}\nVersion 2: {v2}\n\nIs this change COSMETIC or MATERIAL? Explain briefly."

    start = time.perf_counter()
    interaction = client.interactions.create(
        model=MODEL,
        system_instruction=SYSTEM,
        input=prompt,
        store=False,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": 1024,
            "thinking_level": thinking,
        },
    )
    latency_ms = (time.perf_counter() - start) * 1000

    u = interaction.usage
    thought = u.total_thought_tokens or 0

    return {
        "text": interaction.output_text,
        "input_tokens": u.total_input_tokens,
        "output_tokens": u.total_output_tokens,
        "thought_tokens": thought,
        "latency_ms": latency_ms,
        "cost": cost_usd(u.total_input_tokens, u.total_output_tokens, thought),
        "status": interaction.status,
    }



r = classify(V1, V2)

print(r["text"])
print(f"\ntokens in/out/think = {r['input_tokens']}/{r['output_tokens']}/{r['thought_tokens']}")
print(f"latency={r['latency_ms']:.0f} ms  cost=${r['cost']:.6f}  status={r['status']}")

# print("\n===== TEMPERATURE EXPERIMENT =====")

# for temp in [0, 1]:
#     print(f"\n--- temperature={temp} ---")
#     for i in range(3):
#         r = classify(V1, V2, temperature=temp)
#         first_line = r["text"].strip().splitlines()[0]
#         print(f"run {i+1}: {first_line}  ({r['latency_ms']:.0f} ms, think={r['thought_tokens']}, ${r['cost']:.6f})")
#         time.sleep(5)

print("\n===== STATELESSNESS EXPERIMENT =====")

def ask(input_, **kwargs):
    return client.interactions.create(
        model=MODEL,
        input=input_,
        generation_config={"temperature": 0, "thinking_level": "minimal"},
        **kwargs,
    )

a = ask("Remember: our connector is called RTP-Bridge.", store=False)
print("A:", a.output_text)

b = ask("What is our connector called?", store=False)
print("B:", b.output_text)
print("B input tokens:", b.usage.total_input_tokens)

history = [
    # Step 1: what YOU said in call A
    {"type": "user_input",
     "content": [{"type": "text", "text": "Remember: our connector is called RTP-Bridge."}]},

    # Step 2: what the MODEL replied in call A
    {"type": "model_output",
     "content": [{"type": "text", "text": a.output_text}]},

    # Step 3: your new question
    {"type": "user_input",
     "content": [{"type": "text", "text": "What is our connector called?"}]},
]

c1 = ask(history, store=False)
print("\nC1:", c1.output_text)
print("C1 input tokens:", c1.usage.total_input_tokens)

a2 = ask("Remember: our connector is called RTP-Bridge.")
c2 = ask("What is our connector called?", previous_interaction_id=a2.id)
print("\nC2:", c2.output_text)
print("C2 input tokens:", c2.usage.total_input_tokens)

# print("\nC1 usage:", c1.usage)
# print("C2 usage:", c2.usage)

print("\n===== COSMETIC TEST =====")
r = classify(
    "The receiving bank shall return a rejected payment within 30 calendar days.",
    "The receiving bank is required to return a rejected payment within thirty (30) calendar days.",
)
print(r["text"])