
from ollama import chat

MODEL = "huihui_ai/qwen3-abliterated:latest"

messages = [
    {
        "role": "system",
        "content": (
            "You are CAEL, a helpful personal AI assistant. "
            "Respond clearly and concisely unless the user "
            "requests a detailed explanation."
            "Do not use emoji"
        )
    }
]

print("CAEL is online. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ("exit", "quit"):
        break

    if not user_input:
        continue

    messages.append({
        "role": "user",
        "content": user_input
    })

    # Keep the system prompt + last 20 messages
    if len(messages) > 21:
        messages = [messages[0]] + messages[-20:]

    stream = chat(
        model=MODEL,
        messages=messages,
        stream=True,
        think=False,
        keep_alive=-1,
        options={
            "temperature": 0.5,
            "num_ctx": 8192,
            "num_predict": 512
        }
    )

    print("\nCAEL: ", end="", flush=True)

    full_response = ""

    for chunk in stream:
        text = chunk.message.content or ""
        print(text, end="", flush=True)
        full_response += text

    print("\n")

    messages.append({
        "role": "assistant",
        "content": full_response
    })
