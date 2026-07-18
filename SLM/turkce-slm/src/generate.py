import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.models.transformer import MiniGPT

def generate(model, start_tokens, vocab, max_new_tokens=50, temperature=0.8):
    model.eval()
    inv_vocab = {v: k for k, v in vocab.items()}
    tokens = start_tokens[:]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor([tokens], dtype=torch.long)
            logits = model(x)
            next_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            tokens.append(next_token)
            if next_token == vocab.get("<EOS>", -1):
                break

    return [inv_vocab.get(t, "<UNK>") for t in tokens if t not in [vocab.get("<PAD>", 0), vocab.get("<BOS>", 0)]]

if __name__ == "__main__":
    print("=" * 70)
    print("  SLM TOKENIZER TEST")
    print("=" * 70)

    from src.preprocessing import clean_text, simple_tokenize, build_vocab, encode, decode

    sample_texts = [
        "Turkce yapay zeka projeleri cok ilginc.",
        "Dogal dil isleme uzerinde calisiyorum.",
        "Makine ogrenmesi ile veri analizi yapmak mumkun.",
        "Derin ogrenme modelleri cok guclu.",
        "Turkce dil modeli gelistirmek zor ama eglenceli.",
    ]

    cleaned = [clean_text(t) for t in sample_texts]
    vocab = build_vocab(cleaned, min_freq=1)
    inv_vocab = {v: k for k, v in vocab.items()}

    print(f"    Sozluk boyutu: {len(vocab)}")
    print(f"    Ornek: '{cleaned[0]}'")
    ids = encode(cleaned[0], vocab)
    print(f"    Encode: {ids}")
    decoded = decode(ids, inv_vocab)
    print(f"    Decode: {decoded}")
