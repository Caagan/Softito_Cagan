import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("  NAIVE BAYES — 20 Newsgroups Metin Sınıflandırma")
print("=" * 70)

print("\n[PROJE] Haber Grupları Konu Sınıflandırması")
print("-" * 50)

from sklearn.datasets import fetch_20newsgroups

categories = ["sci.space", "rec.sport.baseball", "comp.graphics", "talk.politics.misc"]
train_data = fetch_20newsgroups(subset="train", categories=categories, random_state=42)
test_data = fetch_20newsgroups(subset="test", categories=categories, random_state=42)

print(f"    Eğitim: {len(train_data.data)} metin")
print(f"    Test: {len(test_data.data)} metin")
print(f"    Sınıflar: {train_data.target_names}")

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")
X_train = tfidf.fit_transform(train_data.data)
X_test = tfidf.transform(test_data.data)
y_train = train_data.target
y_test = test_data.target

nb = MultinomialNB(alpha=0.1)
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n    TF-IDF + MultinomialNB Doğruluk: {acc:.4f}")
print(f"\n    Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=train_data.target_names))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Naive Bayes — 20 Newsgroups Sınıflandırması", fontsize=14)

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=train_data.target_names, cmap="Blues", ax=axes[0], xticks_rotation=45)
axes[0].set_title("TF-IDF + MultinomialNB")

alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
alpha_scores = []
for a in alphas:
    nb_temp = MultinomialNB(alpha=a)
    nb_temp.fit(X_train, y_train)
    alpha_scores.append(accuracy_score(y_test, nb_temp.predict(X_test)))

axes[1].plot(alphas, alpha_scores, "o-", color="#3B8BD4", linewidth=2)
axes[1].set_xlabel("Alpha Değeri")
axes[1].set_ylabel("Doğruluk")
axes[1].set_title("Alpha Optimizasyonu")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("c:\\Users\\cagan\\Downloads\\softitoprojelerim\\proje-reposu\\MachineLearning\\Supervised\\08-naive-bayes\\01_newsgroups.png", dpi=120)
plt.close()

feature_names = np.array(tfidf.get_feature_names_out())
print(f"\n    En Önemli Kelimeler (Sınıf Bazlı):")
for i, sinif in enumerate(train_data.target_names):
    top10 = feature_names[nb.coef_[i].argsort()[-10:][::-1]]
    print(f"      {sinif:25s}: {', '.join(top10)}")

print(f"\n{'='*70}")
print(f"  TAMAMLANDI — Doğruluk={acc:.4f}")
print(f"{'='*70}")
