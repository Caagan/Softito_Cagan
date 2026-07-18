import re

def turkish_lower(text):
    text = text.replace("I", "ı").replace("İ", "i")
    return text.lower()

def clean_text(text):
    text = turkish_lower(text)
    text = re.sub(r"<ref>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\[\[.*?\|", "", text)
    text = re.sub(r"\[\[", "", text)
    text = re.sub(r"\]\]", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Zçğıöşüâîûê0-9\s.,!?;:'-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def simple_tokenize(text):
    return text.split()

def build_vocab(texts, min_freq=2):
    freq = {}
    for text in texts:
        for token in simple_tokenize(text):
            freq[token] = freq.get(token, 0) + 1

    vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
    idx = 4
    for token, count in sorted(freq.items(), key=lambda x: -x[1]):
        if count >= min_freq and token not in vocab:
            vocab[token] = idx
            idx += 1

    return vocab

def encode(text, vocab, max_len=512):
    tokens = simple_tokenize(text)
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens[:max_len]]
    ids = [vocab["<BOS>"]] + ids + [vocab["<EOS>"]]
    return ids

def decode(ids, inv_vocab):
    tokens = []
    for i in ids:
        if i in inv_vocab:
            tokens.append(inv_vocab[i])
    return " ".join(tokens)

if __name__ == "__main__":
    ornek = "Bu bir Turkce metin on isleme ornegidir."
    temiz = clean_text(ornek)
    print(f"    Orijinal : {ornek}")
    print(f"    Temiz    : {temiz}")
    tokens = simple_tokenize(temiz)
    print(f"    Tokenlar : {tokens}")
