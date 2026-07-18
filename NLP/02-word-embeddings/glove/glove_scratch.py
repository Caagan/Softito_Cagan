import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("  GLOVE — Global Vectors for Word Representation")
print("=" * 70)

print("\n[PROJE] GloVe Sifirdan Implementasyonu (Ko-Matris)")
print("-" * 50)

corpus = [
    "makine ogrenmesi yapay zeka temelidir",
    "derin ogrenme sinir aglari kullanir",
    "dogal dil isleme metin analiz eder",
    "veri bilimi istatistik ve kodlama gerektirir",
    "yapay zeka gelecegi degistirecek",
    "makine ogrenmesi ile tahmin yapilabilir",
    "sinir aglari derin ogrenme temelidir",
    "istatistik veri bilimi icin onemlidir",
]

words = " ".join(corpus).split()
vocab = list(set(words))
word2idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)

print(f"    Sozluk boyutu: {vocab_size}")

def build_cooccurrence(corpus, window_size=2):
    cooc = np.zeros((vocab_size, vocab_size))
    for sent in corpus:
        tokens = sent.split()
        for i, center in enumerate(tokens):
            ci = word2idx[center]
            for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
                if i != j:
                    cj = word2idx[tokens[j]]
                    dist = abs(i - j)
                    cooc[ci, cj] += 1.0 / dist
    return cooc

cooc_matrix = build_cooccurrence(corpus)
print(f"    Ko-matris boyutu: {cooc_matrix.shape}")
print(f"    Sifir olmayan: {np.count_nonzero(cooc_matrix)}")

import torch
import torch.nn as nn

class GloVeModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=10):
        super().__init__()
        self.W = nn.Embedding(vocab_size, embed_dim)
        self.W_tilde = nn.Embedding(vocab_size, embed_dim)
        self.b = nn.Embedding(vocab_size, 1)
        self.b_tilde = nn.Embedding(vocab_size, 1)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.W_tilde.weight)

    def forward(self, i, j, x):
        wi = self.W(i)
        wj = self.W_tilde(j)
        bi = self.b(i)
        bj = self.b_tilde(j)
        dot = torch.sum(wi * wj, dim=1, keepdim=True)
        return (dot + bi + bj - torch.log(x.unsqueeze(1))) ** 2

def weighting_func(x, x_max=100, alpha=0.75):
    return (x / x_max) ** alpha * (x < x_max).float() + (x >= x_max).float()

embed_dim = 10
model = GloVeModel(vocab_size, embed_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

non_zero = np.argwhere(cooc_matrix > 0)
i_idx = torch.tensor(non_zero[:, 0], dtype=torch.long)
j_idx = torch.tensor(non_zero[:, 1], dtype=torch.long)
x_val = torch.tensor([cooc_matrix[non_zero[k, 0], non_zero[k, 1]] for k in range(len(non_zero))], dtype=torch.float)
weights = weighting_func(x_val)

print(f"\n    Egitim basliyor (100 epoch)...")
losses = []
for epoch in range(100):
    pred = model(i_idx, j_idx, x_val)
    loss = (weights * pred).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

print(f"    Final kayip: {loss.item():.4f}")

embeddings = (model.W.weight.data + model.W_tilde.weight.data).numpy() / 2

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("GloVe — Global Word Vectors", fontsize=14)

axes[0].plot(losses, "b-", linewidth=2)
axes[0].set_title("Egitim Kayibi")
axes[0].set_xlabel("Epoch")
axes[0].grid(alpha=0.3)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
coords = pca.fit_transform(embeddings)

for i, word in enumerate(vocab):
    axes[1].scatter(coords[i, 0], coords[i, 1], color="#E24B4A", s=50)
    axes[1].annotate(word, (coords[i, 0]+0.01, coords[i, 1]+0.01), fontsize=8)
axes[1].set_title("2D Kelime Uzayi (PCA)")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(r"c:\Users\cagan\Downloads\softitoprojelerim\proje-reposu\NLP\02-word-embeddings\glove\01_glove_embedding.png", dpi=120)
plt.close()

print(f"\n{'='*70}")
print(f"  TAMAMLANDI")
print(f"{'='*70}")
