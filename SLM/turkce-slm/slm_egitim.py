import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import urllib.request
import os
import re

print("=" * 70)
print("  SLM — Türkçe Karakter Seviyeli Dil Modeli (Decoder-Only Transformer)")
print("=" * 70)

print("\n[PROJE] Türkçe Vikipedi Alt Kümesi — Gerçek Metin Üzerinde SLM Eğitimi")
print("-" * 50)

DATA_URL = "https://huggingface.co/datasets/ancs21/turkish-wikipedia/resolve/main/data/train-00000-of-00001.parquet"
DATA_DIR = r"c:\Users\cagan\Downloads\softitoprojelerim\proje-reposu\SLM\turkce-slm\data"
DATA_FILE = os.path.join(DATA_DIR, "turkish_wiki.txt")

def indir_veri():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(DATA_FILE):
        print("    Veri zaten mevcut.")
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return f.read()

    print("    Türkçe Wikipedia verisi indiriliyor...")
    parquet_path = os.path.join(DATA_DIR, "train.parquet")
    try:
        urllib.request.urlretrieve(DATA_URL, parquet_path)
        print("    Parquet dosyası indirildi, işleniyor...")
        import pandas as pd
        df = pd.read_parquet(parquet_path)
        text_col = df.columns[0]
        text = "\n".join(df[text_col].astype(str).tolist())
    except Exception:
        print("    HuggingFace bağlantısı başarısız, alternatif kaynak deneniyor...")
        alt_urls = [
            "https://raw.githubusercontent.com/stefan-it/turkish-bert/master/data/ocr/tr.txt",
            "https://raw.githubusercontent.com/turkish-nlp-corpus/turkish-nlp-corpus/master/corpus.txt",
        ]
        text = ""
        for url in alt_urls:
            try:
                tmp = os.path.join(DATA_DIR, "raw.txt")
                urllib.request.urlretrieve(url, tmp)
                with open(tmp, "r", encoding="utf-8") as f:
                    text = f.read()
                if len(text) > 10000:
                    break
            except Exception:
                continue

        if len(text) < 10000:
            print("    Çevrimdışı kaynaklar başarısız, HuggingFace datasets deneniyor...")
            try:
                from datasets import load_dataset
                ds = load_dataset("tuncsinvipk/turkish-wikipedia", split="train[:20000]", trust_remote_code=True)
                text = "\n".join([x[list(x.keys())[0]] for x in ds])
            except Exception:
                print("    Hata: Yeterli Türkçe veri indirilemedi.")
                print("    Yerel bir Türkçe dosya (.txt) kullanabilirsiniz:")
                print(f"    Dosyayı şuraya koyun: {DATA_FILE}")
                text = ""

    with open(DATA_FILE, "w", encoding="utf-8") as f:
        f.write(text)
    return text

VERI = indir_veri()

if len(VERI) < 1000:
    print("\n    [UYARI] Yeterli veri bulunamadı, model eğitilemez.")
    print("    Lütfen bir Türkçe .txt dosyasını şu yola koyun:")
    print(f"    {DATA_FILE}")
    import sys; sys.exit(1)

# Limit to reasonable size for local training
MAX_CHARS = 500_000
if len(VERI) > MAX_CHARS:
    VERI = VERI[:MAX_CHARS]
    print(f"    Veri {MAX_CHARS:,} karakterle sınırlandırıldı.")
else:
    print(f"    Toplam karakter sayısı: {len(VERI):,}")

karakterler = sorted(list(set(VERI)))
kar_to_idx = {k: i for i, k in enumerate(karakterler)}
idx_to_kar = {i: k for k, i in kar_to_idx.items()}
VOCAB_SIZE = len(karakterler)

print(f"\n    Karakter Seti Boyutu : {VOCAB_SIZE}")
print(f"    Benzersiz Karakterler: {len(karakterler)}")

# Sample characters for display
ornek_karakterler = ''.join(karakterler[:40])
print(f"    Örnek Karakterler    : {ornek_karakterler}")

SEQ_LEN = 128
BATCH_SIZE = 32
HIDDEN_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 4
EPOCHS = 30
LR = 3e-4

class KarakterDataset(Dataset):
    def __init__(self, veri, seq_len):
        self.kodlar = [kar_to_idx[k] for k in veri]
        self.seq_len = seq_len

    def __len__(self):
        return len(self.kodlar) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.kodlar[idx:idx+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.kodlar[idx+1:idx+self.seq_len+1], dtype=torch.long)
        return x, y

dataset = KarakterDataset(VERI, SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"    Toplam Örnek Sayısı : {len(dataset):,}")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, seq_len)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.seq_len = seq_len

    def forward(self, x):
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        embeds = self.pos_enc(self.embedding(x))
        output = self.transformer(embeds, embeds, tgt_mask=mask)
        return self.fc_out(output)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"    Cihaz: {device}")

model = MiniGPT(VOCAB_SIZE, HIDDEN_DIM, NUM_HEADS, NUM_LAYERS, SEQ_LEN).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"    Toplam Parametre     : {total_params:,}")

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

print(f"\n    Eğitim Başlıyor ({EPOCHS} epoch, {len(dataset):,} örnek)...")
print(f"    {'Epoch':>6}  {'Kayıp':>8}  {'İlerleme'}")
print(f"    {'─'*50}")

train_losses = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    batch_count = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1

    scheduler.step()
    avg_loss = total_loss / batch_count
    train_losses.append(avg_loss)
    progress = "█" * (avg_loss / max(train_losses) * 20) if max(train_losses) > 0 else ""
    if epoch % 5 == 0 or epoch == 1:
        print(f"    {epoch:>6}  {avg_loss:>8.4f}  {progress}")

def uret(model, baslangic_metni, uzunluk=300, temperature=0.7):
    model.eval()
    giris = torch.tensor([[kar_to_idx.get(k, 0) for k in baslangic_metni]], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(uzunluk):
            giris_son = giris[:, -SEQ_LEN:]
            logits = model(giris_son)
            son_logits = logits[:, -1, :] / temperature
            olasilik = torch.softmax(son_logits, dim=-1)
            sonraki = torch.multinomial(olasilik, 1)
            giris = torch.cat([giris, sonraki], dim=1)

    uretilen = "".join([idx_to_kar.get(i.item(), "?") for i in giris[0]])
    return uretilen

print(f"\n    Modelle Metin Üretimi:")
print(f"    {'─'*50}")

ornekler = [
    "Yapay zeka",
    "Bilgisayar bilimi",
    "İstanbul",
    "Türkçe doğal dil",
]

for baslangic in ornekler:
    sonuc = uret(model, baslangic, uzunluk=150)
    print(f"    Başlangıç: '{baslangic}'")
    print(f"    Üretilen : {sonuc[:200]}")
    print()

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_losses)+1), train_losses, "o-", color="#3B8BD4", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Kayıp")
plt.title("SLM Eğitim Kayıp Eğrisi — Türkçe Dil Modeli")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(r"c:\Users\cagan\Downloads\softitoprojelerim\proje-reposu\SLM\turkce-slm\01_slm_kayip.png", dpi=120)
plt.close()

print(f"\n{'='*70}")
print(f"  TAMAMLANDI — Türkçe SLM eğitimi ve metin üretimi tamamlandı.")
print(f"{'='*70}")
