import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("  FASHION-MNIST CNN — Uçtan Uca Görüntü Sınıflandırma")
print("=" * 70)

print("\n[ADIM 1] Veri Setini Yükleme")
print("-" * 50)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

BATCH_SIZE = 128
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

siniflar = ["Tişört", "Pantolon", "Kazak", "Elbise", "Ceket",
            "Sandalet", "Gömlek", "Sneaker", "Çanta", "Bot"]

print(f"    Eğitim verisi   : {len(train_dataset):,} görüntü")
print(f"    Test verisi     : {len(test_dataset):,} görüntü")
print(f"    Görüntü boyutu  : 28x28 piksel (gri tonlu)")
print(f"    Sınıf sayısı    : {len(siniflar)}")

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("Fashion-MNIST Örnek Görüntüler", fontsize=14)
for i, ax in enumerate(axes.flat):
    img, label = train_dataset[i]
    ax.imshow(img.squeeze(), cmap="gray")
    ax.set_title(siniflar[label], fontsize=10)
    ax.axis("off")
plt.tight_layout()
plt.savefig("c:\\Users\\cagan\\Downloads\\softitoprojelerim\\proje-reposu\\DeepLearning\\CNN\\01-fashion-mnist-cnn\\01_ornek_goruntuler.png", dpi=120)
plt.close()

print(f"\n[ADIM 2] CNN Model Mimarisi")
print("-" * 50)

class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FashionCNN().to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"    Cihaz              : {device}")
print(f"    Toplam Parametre   : {total_params:,}")
print(f"\n    Model Mimarisi:")
print(f"    Conv2d(1→32) → BN → ReLU → MaxPool")
print(f"    Conv2d(32→64) → BN → ReLU → MaxPool")
print(f"    Conv2d(64→128) → BN → ReLU → MaxPool")
print(f"    Flatten → Linear(1152→256) → Dropout → Linear(256→128) → Linear(128→10)")

print(f"\n[ADIM 3] Eğitim")
print("-" * 50)

EPOCHS = 15
LR = 0.001

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

print(f"    {'Epoch':>6}  {'Train Kayıp':>12}  {'Test Kayıp':>12}  {'Train Acc':>10}  {'Test Acc':>10}")
print(f"    {'─'*55}")

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

    model.eval()
    test_loss, test_correct, test_total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    train_loss /= len(train_loader)
    test_loss /= len(test_loader)
    train_acc = 100. * train_correct / train_total
    test_acc = 100. * test_correct / test_total

    history["train_loss"].append(train_loss)
    history["test_loss"].append(test_loss)
    history["train_acc"].append(train_acc)
    history["test_acc"].append(test_acc)

    scheduler.step(test_loss)

    if epoch % 3 == 0 or epoch == 1:
        print(f"    {epoch:>6}  {train_loss:>12.4f}  {test_loss:>12.4f}  {train_acc:>9.2f}%  {test_acc:>9.2f}%")

final_test_acc = history["test_acc"][-1]
print(f"\n    Final Test Doğruluğu: {final_test_acc:.2f}%")

print(f"\n[ADIM 4] Sınıflandırma Raporu")
print("-" * 50)
print(classification_report(all_labels, all_preds, target_names=siniflar))

print(f"\n[ADIM 5] Görselleştirme")
print("-" * 50)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Fashion-MNIST CNN Eğitim Sonuçları", fontsize=14)

axes[0, 0].plot(history["train_loss"], label="Eğitim", color="#3B8BD4", linewidth=2)
axes[0, 0].plot(history["test_loss"], label="Test", color="#E24B4A", linewidth=2)
axes[0, 0].set_title("Kayıp (Loss) Eğrisi")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Kayıp")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(history["train_acc"], label="Eğitim", color="#3B8BD4", linewidth=2)
axes[0, 1].plot(history["test_acc"], label="Test", color="#E24B4A", linewidth=2)
axes[0, 1].set_title("Doğruluk (Accuracy) Eğrisi")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Doğruluk (%)")
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=siniflar, yticklabels=siniflar, ax=axes[1, 0])
axes[1, 0].set_title("Confusion Matrix")
axes[1, 0].set_xlabel("Tahmin")
axes[1, 0].set_ylabel("Gerçek")
axes[1, 0].tick_params(axis="x", rotation=45)
axes[1, 0].tick_params(axis="y", rotation=0)

sinif_dogruluk = cm.diagonal() / cm.sum(axis=1) * 100
bars = axes[1, 1].barh(siniflar, sinif_dogruluk, color="#1D9E75")
axes[1, 1].bar_label(bars, fmt="%.1f%%", padding=4)
axes[1, 1].set_xlim(0, 110)
axes[1, 1].set_title("Sınıf Bazlı Doğruluk")
axes[1, 1].set_xlabel("Doğruluk (%)")
axes[1, 1].grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("c:\\Users\\cagan\\Downloads\\softitoprojelerim\\proje-reposu\\DeepLearning\\CNN\\01-fashion-mnist-cnn\\02_egitim_sonuclari.png", dpi=120)
plt.close()

fig, axes = plt.subplots(3, 5, figsize=(15, 9))
fig.suptitle("Gerçek vs Tahmin — Rastgele Test Görüntüleri", fontsize=14)
model.eval()
indices = np.random.choice(len(test_dataset), 15, replace=False)
for i, ax in enumerate(axes.flat):
    img, label = test_dataset[indices[i]]
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device)).argmax().item()
    renk = "green" if label == pred else "red"
    ax.imshow(img.squeeze(), cmap="gray")
    ax.set_title(f"G:{siniflar[label]}\nT:{siniflar[pred]}", fontsize=9, color=renk)
    ax.axis("off")
plt.tight_layout()
plt.savefig("c:\\Users\\cagan\\Downloads\\softitoprojelerim\\proje-reposu\\DeepLearning\\CNN\\01-fashion-mnist-cnn\\03_tahmin_ornekleri.png", dpi=120)
plt.close()

print(f"    Görseller kaydedildi.")
print(f"\n{'='*70}")
print(f"  TAMAMLANDI — Fashion-MNIST CNN sınıflandırma ({final_test_acc:.2f}% doğruluk)")
print(f"{'='*70}")
