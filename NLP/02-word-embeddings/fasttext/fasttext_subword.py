import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import hashlib
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("  FASTTEXT — Subword Embedding")
print("=" * 70)

print("\n[PROJE] FastText Subword Temsili Sifirdan")
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

def get_subwords(word, n_range=(3, 6)):
    subwords = [f"<{word}>"]
    for n in range(n_range[0], n_range[1] + 1):
        for i in range(len(word) - n + 1):
            subwords.append(word[i:i + n])
    return subwords

print(f"\n    Ornek subword'-ler:")
for word in ["ogrenmesi", "yapay"]:
    sw = get_subwords(word)
    print(f"    '{word}' -> {sw[:8]}...")

all_subwords = set()
for word in vocab:
    all_subwords.update(get_subwords(word))
all_subwords = list(all_subwords)
subword2idx = {s: i for i, s in enumerate(all_subwords)}
num_subwords = len(all_subwords)

print(f"    Toplam subword: {num_subwords}")

def get_hash_embeddings(word, embed_dim=10):
    hashes = []
    for sw in get_subwords(word):
        h = int(hashlib.md5(sw.encode()).hexdigest(), 16) % 10000
        hashes.append(h)
    np.random.seed(hashes[0] % (2**31))
    return np.random.randn(embed_dim).astype(np.float32)

embed_dim = 10
embeddings = np.zeros((vocab_size, embed_dim), dtype=np.float32)

for word in vocab:
    embeddings[word2idx[word]] = get_hash_embeddings(word, embed_dim)

print(f"    Embedding boyutu: {embeddings.shape}")

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
coords = pca.fit_transform(embeddings)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("FastText — Subword Embedding", fontsize=14)

for i, word in enumerate(vocab):
    axes[0].scatter(coords[i, 0], coords[i, 1], color="#1D9E75", s=50)
    axes[0].annotate(word, (coords[i, 0]+0.01, coords[i, 1]+0.01), fontsize=8)
axes[0].set_title("FastText Subword Uzayi (PCA)")
axes[0].grid(alpha=0.3)

subword_lengths = Counter()
for word in vocab:
    for sw in get_subwords(word):
        subword_lengths[len(sw)] += 1

lengths = sorted(subword_lengths.keys())
counts = [subword_lengths[l] for l in lengths]
axes[1].bar(lengths, counts, color="#3B8BD4")
axes[1].set_title("Subword Uzunluk Dagilimi")
axes[1].set_xlabel("Uzunluk")
axes[1].set_ylabel("Frekans")

plt.tight_layout()
plt.savefig(r"c:\Users\cagan\Downloads\softitoprojelerim\proje-reposu\NLP\02-word-embeddings\fasttext\01_fasttext_subword.png", dpi=120)
plt.close()

print(f"\n{'='*70}")
print(f"  TAMAMLANDI")
print(f"{'='*70}")
