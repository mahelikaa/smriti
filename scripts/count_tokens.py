import anthropic

client = anthropic.Anthropic()

sentences = {
    "English": "My best friend Riya's birthday is on 12 March.",
    "Romanized Hinglish": "Meri best friend Riya ka birthday 12 March ko hai.",
    "Devanagari Hindi": "मेरी सबसे अच्छी दोस्त रिया का जन्मदिन 12 मार्च को है।",
}

for language, text in sentences.items():
    response = client.messages.count_tokens(
        model="claude-opus-5-5",
        messages=[{"role": "user", "content": text}],
    )

    print(f"{language}: {response.input_tokens} tokens")