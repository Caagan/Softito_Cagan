import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("  TF-IDF — Metin Sınıflandırma & Boyut İndirgeme")
print("=" * 70)

print("\n[PROJE] 20 Newsgroups — TF-IDF + Boyut İndirgeme Analizi")
print("-" * 50)

categories = ["sci.space", "rec.sport.baseball", "comp.graphics", "talk.politics.misc"]
train_data = fetch_20newsgroups(subset="train", categories=categories, random_state=42)
test_data = fetch_20newsgroups(subset="test", categories=categories, random_state=42)

print(f"    Eğitim: {len(train_data.data)} metin")
print(f"    Test: {len(test_data.data)} metin")
print(f"    Sınıflar: {train_data.target_names}")

X_train_text = train_data.data
X_test_text = test_data.data
y_train = train_data.target
y_test = test_data.target

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

print(f"\n    TF-IDF matris boyutu: {X_train_tfidf.shape}")

model_full = LogisticRegression(max_iter=1000, random_state=42)
model_full.fit(X_train_tfidf, y_train)
y_pred_full = model_full.predict(X_test_tfidf)
acc_full = accuracy_score(y_test, y_pred_full)

print(f"    Tam Boyut Doğruluk: {acc_full:.4f}")

boyutlar = [100, 200, 300, 500]
sonuclar = []
print(f"\n    {'Boyut':>8}  {'Doğruluk':>10}  {'Varyans':>10}  {'Sıkışma':>10}")
print(f"    {'─'*45}")

for n in boyutlar:
    svd = TruncatedSVD(n_components=n, random_state=42)
    X_train_svd = svd.fit_transform(X_train_tfidf)
    X_test_svd = svd.transform(X_test_tfidf)
    varyans = svd.explained_variance_ratio_.sum() * 100

    model_svd = LogisticRegression(max_iter=1000, random_state=42)
    model_svd.fit(X_train_svd, y_train)
    y_pred_svd = model_svd.predict(X_test_svd)
    acc_svd = accuracy_score(y_test, y_pred_svd)
    sikistirme = (1 - n / X_train_tfidf.shape[1]) * 100

    sonuclar.append({"boyut": n, "dogruluk": acc_svd, "varyans": varyans, "sikistirme": sikistirme})
    print(f"    {n:>8}  {acc_svd:>10.4f}  %{varyans:>8.1f}  %{sikistirme:>8.1f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("TF-IDF — Boyut İndirgeme Analizi", fontsize=14)

boyutlar_plot = [s["boyut"] for s in sonuclar]
dogruluklar = [s["dogruluk"] for s in sonuclar]
varyanslar = [s["varyans"] for s in sonuclar]

axes[0].plot(boyutlar_plot, dogruluklar, "o-", color="#3B8BD4", linewidth=2, markersize=8)
axes[0].axhline(acc_full, color="#E24B4A", linestyle="--", label=f"Tam Boyut ({acc_full:.4f})")
axes[0].set_xlabel("SVD Boyutu")
axes[0].set_ylabel("Doğruluk")
axes[0].set_title("Boyut vs Doğruluk")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].bar([str(b) for b in boyutlar_plot], varyanslar, color="#1D9E75", width=0.5)
axes[1].set_xlabel("SVD Boyutu")
axes[1].set_ylabel("Açıklanan Varyans (%)")
axes[1].set_title("Açıklanan Varyans Oranı")
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("c:\\Users\\cagan\\Downloads\\softitoprojelerim\\proje-reposu\\NLP\\01-tf-idf\\01_tfidf_analiz.png", dpi=120)
plt.close()

best = max(sonuclar, key=lambda x: x["dogruluk"])
print(f"\n    En İyi: {best['boyut']} boyut → Doğruluk: {best['dogruluk']:.4f}")
print(f"    Sıkışma: %{best['sikistirma']:.1f}")

print(f"\n{'='*70}")
print(f"  TAMAMLANDI")
print(f"{'='*70}")
