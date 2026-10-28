import anthropic

client = anthropic.Anthropic()
message = client.messages.create(
    model="claude-opus-5-5",
    max_tokens=1000,
    messages=[{"role": "user", "content": "Say hi in Hinglish in one short sentence."}],
)

text = next(block.text for block in message.content if block.type == "text")
print(text)
print("stop_reason:", message.stop_reason)
print("usage:", message.usage)