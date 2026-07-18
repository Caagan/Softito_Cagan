import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.models.transformer import MiniGPT
from src.preprocessing import clean_text, simple_tokenize, build_vocab, encode, decode

def main():
    print("=" * 50)
    print("  TURKCE SLM — Demo")
    print("=" * 50)

    sample_texts = [
        "Turkce dil modeli ile metin uretimi yapilabilir.",
        "Yapay zeka teknolojileri hizli bir sekilde gelisiyor.",
        "Dogal dil isleme cok onemli bir alandir.",
    ]

    vocab = build_vocab([clean_text(t) for t in sample_texts], min_freq=1)
    inv_vocab = {v: k for k, v in vocab.items()}

    model = MiniGPT(vocab_size=len(vocab), embed_dim=64, num_heads=2, num_layers=2, max_len=64)
    print(f"    Model yuklendi. Parametre: {sum(p.numel() for p in model.parameters()):,}")

    while True:
        text = input("\n    Metin girin (cikis icin 'q'): ").strip()
        if text.lower() == "q":
            break
        cleaned = clean_text(text)
        ids = encode(cleaned, vocab)
        print(f"    Temizlenmis: {cleaned}")
        print(f"    Tokenler: {simple_tokenize(cleaned)}")
        print(f"    ID'ler: {ids}")

if __name__ == "__main__":
    main()
