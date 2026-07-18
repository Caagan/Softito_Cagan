import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("  ATTENTION — BiLSTM + Bahdanau Attention ile Duygu Analizi")
print("=" * 70)

print("\n[PROJE] IMDB Film Yorumları — Duygu Analizi")
print("-" * 50)

try:
    from datasets import load_dataset
    dataset = load_dataset("imdb", trust_remote_code=True)
    train_texts = dataset["train"]["text"][:2000]
    train_labels = dataset["train"]["label"][:2000]
    test_texts = dataset["test"]["text"][:500]
    test_labels = dataset["test"]["label"][:500]
    print(f"    HuggingFace IMDB yüklendi")
except Exception:
    print("    HuggingFace bağlantısı yok, sklearn kullanılıyor...")
    from sklearn.datasets import fetch_20newsgroups
    data_pos = fetch_20newsgroups(subset="train", categories=["rec.sport.baseball"], random_state=42)
    data_neg = fetch_20newsgroups(subset="train", categories=["talk.politics.misc"], random_state=42)
    train_texts = data_pos.data[:1000] + data_neg.data[:1000]
    train_labels = [1] * 1000 + [0] * 1000
    test_texts = data_pos.data[1000:1250] + data_neg.data[1000:1250]
    test_labels = [1] * 250 + [0] * 250

print(f"    Eğitim: {len(train_texts)}, Test: {len(test_texts)}")

tum_kelimeler = " ".join(train_texts).lower().split()
kelime_sayaclari = {}
for k in tum_kelimeler:
    k = k.strip(".,!?;:\"'()[]{}")
    if k and len(k) > 1:
        kelime_sayaclari[k] = kelime_sayaclari.get(k, 0) + 1

MAX_VOCAB = 10000
sorted_kelimeler = sorted(kelime_sayaclari.items(), key=lambda x: x[1], reverse=True)
kelime_to_idx = {"<PAD>": 0, "<UNK>": 1}
for kelime, _ in sorted_kelimeler[:MAX_VOCAB - 2]:
    kelime_to_idx[kelime] = len(kelime_to_idx)

print(f"    Dağarcık: {len(kelime_to_idx)}")

MAX_LEN = 100

def tokenize(metin):
    return [kelime_to_idx.get(k.lower().strip(".,!?;:\"'()[]{}"), 1) for k in metin.split() if k.strip(".,!?;:\"'()[]{}")][:MAX_LEN]

class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = tokenize(self.texts[idx])
        uzunluk = len(tokens) if tokens else 1
        return torch.tensor(tokens if tokens else [0], dtype=torch.long), uzunluk, self.labels[idx]

def collate_fn(batch):
    yorumlar, uzunluklar, etiketler = zip(*batch)
    uzunluklar = torch.tensor(uzunluklar, dtype=torch.long)
    etiketler = torch.tensor(etiketler, dtype=torch.float)
    yorumlar_padded = pad_sequence(yorumlar, batch_first=True, padding_value=0)
    return yorumlar_padded, uzunluklar, etiketler

train_dataset = SentimentDataset(train_texts, train_labels)
test_dataset = SentimentDataset(test_texts, test_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden_states, mask=None):
        score = self.V(torch.tanh(self.W(hidden_states))).squeeze(-1)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(score, dim=1)
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        return context, weights

class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.attention = BahdanauAttention(hidden_dim * 2)
        self.fc = nn.Sequential(nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        mask = (x != 0)
        context, attn_weights = self.attention(output, mask)
        logits = self.fc(context).squeeze(-1)
        return logits, attn_weights

EMBED_DIM = 128
HIDDEN_DIM = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BiLSTMAttention(len(kelime_to_idx), EMBED_DIM, HIDDEN_DIM).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"    Parametre: {total_params:,}")

EPOCHS = 10
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

print(f"\n    {'Epoch':>6}  {'Train Loss':>12}  {'Test Loss':>12}  {'Train Acc':>10}  {'Test Acc':>10}")
print(f"    {'─'*55}")

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    for x, lengths, y in train_loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x, lengths)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        train_correct += (preds == y).sum().item()
        train_total += y.size(0)

    model.eval()
    test_loss, test_correct, test_total = 0, 0, 0
    all_attn_weights = []
    with torch.no_grad():
        for x, lengths, y in test_loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            logits, attn = model(x, lengths)
            loss = criterion(logits, y)
            test_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            test_correct += (preds == y).sum().item()
            test_total += y.size(0)
            all_attn_weights.append(attn.cpu())

    train_loss /= len(train_loader)
    test_loss /= len(test_loader)
    history["train_loss"].append(train_loss)
    history["test_loss"].append(test_loss)
    history["train_acc"].append(100. * train_correct / train_total)
    history["test_acc"].append(100. * test_correct / test_total)

    if epoch % 2 == 0 or epoch == 1:
        print(f"    {epoch:>6}  {train_loss:>12.4f}  {test_loss:>12.4f}  {history['train_acc'][-1]:>9.2f}%  {history['test_acc'][-1]:>9.2f}%")

final_acc = history["test_acc"][-1]
print(f"\n    Final Test Doğruluk: {final_acc:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("BiLSTM + Bahdanau Attention — Duygu Analizi", fontsize=14)

axes[0].plot(history["train_loss"], label="Eğitim", color="#3B8BD4", linewidth=2)
axes[0].plot(history["test_loss"], label="Test", color="#E24B4A", linewidth=2)
axes[0].set_title("Kayıp Eğrisi")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history["train_acc"], label="Eğitim", color="#3B8BD4", linewidth=2)
axes[1].plot(history["test_acc"], label="Test", color="#E24B4A", linewidth=2)
axes[1].set_title("Doğruluk Eğrisi")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Doğruluk (%)")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("c:\\Users\\cagan\\Downloads\\softitoprojelerim\\proje-reposu\\NLP\\05-attention\\01_attention_analiz.png", dpi=120)
plt.close()

print(f"\n{'='*70}")
print(f"  TAMAMLANDI — Test Doğruluk: {final_acc:.2f}%")
print(f"{'='*70}")
