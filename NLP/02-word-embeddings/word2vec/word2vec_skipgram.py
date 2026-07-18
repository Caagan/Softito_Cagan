import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("  WORD2VEC — Skip-Gram Sifirdan")
print("=" * 70)

print("\n[PROJE] Word2Vec Skip-Gram Implementasyonu")
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
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(vocab)

print(f"    Sozluk boyutu: {vocab_size}")
print(f"    Toplam kelime: {len(words)}")

def create_skipgram_pairs(window_size=2):
    pairs = []
    for sent in [s.split() for s in corpus]:
        for i, center in enumerate(sent):
            for j in range(max(0, i - window_size), min(len(sent), i + window_size + 1)):
                if i != j:
                    pairs.append((word2idx[center], word2idx[sent[j]]))
    return pairs

pairs = create_skipgram_pairs()
print(f"    Skip-Gram ciftleri: {len(pairs)}")

import torch
import torch.nn as nn

class Word2VecSkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim=10):
        super().__init__()
        self.center_embed = nn.Embedding(vocab_size, embed_dim)
        self.context_embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, center, context):
        c = self.center_embed(center)
        ctx = self.context_embed(context)
        return torch.sum(c * ctx, dim=1)

embed_dim = 10
model = Word2VecSkipGram(vocab_size, embed_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

centers = torch.tensor([p[0] for p in pairs])
contexts = torch.tensor([p[1] for p in pairs])

print(f"\n    Egitim basliyor (50 epoch)...")
for epoch in range(50):
    scores = model(centers, contexts)
    loss = torch.mean(torch.clamp(-scores + 1, min=0))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"    Final kayip: {loss.item():.4f}")

embeddings = model.center_embed.weight.data.numpy()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Word2Vec Skip-Gram — Kelime Uzayi", fontsize=14)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
coords = pca.fit_transform(embeddings)

for i, word in enumerate(vocab):
    axes[0].scatter(coords[i, 0], coords[i, 1], color="#3B8BD4", s=50)
    axes[0].annotate(word, (coords[i, 0]+0.01, coords[i, 1]+0.01), fontsize=8)
axes[0].set_title("2D Kelime Uzayi (PCA)")
axes[0].grid(alpha=0.3)

cosine_sim = np.zeros((vocab_size, vocab_size))
for i in range(vocab_size):
    for j in range(vocab_size):
        a, b = embeddings[i], embeddings[j]
        cosine_sim[i, j] = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

im = axes[1].imshow(cosine_sim, cmap="coolwarm")
axes[1].set_xticks(range(vocab_size))
axes[1].set_yticks(range(vocab_size))
axes[1].set_xticklabels(vocab, rotation=90, fontsize=6)
axes[1].set_yticklabels(vocab, fontsize=6)
axes[1].set_title("Cosine Similarity Matrisi")
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.savefig(r"c:\Users\cagan\Downloads\softitoprojelerim\proje-reposu\NLP\02-word-embeddings\word2vec\01_word2vec_skipgram.png", dpi=120)
plt.close()

print(f"\n{'='*70}")
print(f"  TAMAMLANDI")
print(f"{'='*70}")
