import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("  FINANS SINIFLANDIRICI — LoRA ile Fine-Tuning")
print("=" * 70)

print("\n[PROJE] California Housing — Regresyon Fine-Tuning")
print("-" * 50)

housing = fetch_california_housing()
X = housing.data
y = housing.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

train_size = int(0.8 * len(X))
X_train = torch.tensor(X_scaled[:train_size], dtype=torch.float32)
y_train = torch.tensor(y[:train_size], dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_scaled[train_size:], dtype=torch.float32)
y_test = torch.tensor(y[train_size:], dtype=torch.float32).unsqueeze(1)

print(f"    Train: {X_train.shape}, Test: {X_test.shape}")

class HousingModel(torch.nn.Module):
    def __init__(self, in_dim=8, hidden=64, out_dim=1):
        super().__init__()
        self.base = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.head(self.base(x))

class LoRALinear(torch.nn.Module):
    def __init__(self, linear, rank=4, alpha=8):
        super().__init__()
        self.linear = linear
        self.lora_A = torch.nn.Linear(linear.in_features, rank, bias=False)
        self.lora_B = torch.nn.Linear(rank, linear.out_features, bias=False)
        self.scaler = alpha / rank
        torch.nn.init.kaiming_uniform_(self.lora_A.weight)
        torch.nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.linear(x) + self.scaler * self.lora_B(self.lora_A(x))

base_model = HousingModel()
lora_model = HousingModel()
for param in lora_model.base.parameters():
    param.requires_grad = False

lora_layer_0 = LoRALinear(lora_model.base[0], rank=4)
lora_layer_2 = LoRALinear(lora_model.base[2], rank=4)
lora_model.base = torch.nn.Sequential(
    lora_layer_0, lora_model.base[1],
    lora_layer_2, lora_model.base[3]
)

print(f"    Base parametre: {sum(p.numel() for p in base_model.parameters()):,}")
print(f"    LoRA trainable: {sum(p.numel() for p in lora_model.parameters() if p.requires_grad):,}")

optimizer_base = torch.optim.Adam(base_model.parameters(), lr=1e-3)
optimizer_lora = torch.optim.Adam(filter(lambda p: p.requires_grad, lora_model.parameters()), lr=1e-3)
criterion = torch.nn.MSELoss()

print(f"\n    Base model egitimi (50 epoch)...")
base_losses = []
for epoch in range(50):
    pred = base_model(X_train)
    loss = criterion(pred, y_train)
    optimizer_base.zero_grad()
    loss.backward()
    optimizer_base.step()
    base_losses.append(loss.item())

print(f"    LoRA model egitimi (50 epoch)...")
lora_losses = []
for epoch in range(50):
    pred = lora_model(X_train)
    loss = criterion(pred, y_train)
    optimizer_lora.zero_grad()
    loss.backward()
    optimizer_lora.step()
    lora_losses.append(loss.item())

with torch.no_grad():
    base_test = criterion(base_model(X_test), y_test).item()
    lora_test = criterion(lora_model(X_test), y_test).item()

print(f"\n    Base Test MSE: {base_test:.4f}")
print(f"    LoRA Test MSE: {lora_test:.4f}")
print(f"    MSE farki: {abs(base_test - lora_test):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("LoRA Fine-Tuning — Finans Siniflandirici", fontsize=14)

axes[0].plot(base_losses, label="Base Model", linewidth=2)
axes[0].plot(lora_losses, label="LoRA Model", linewidth=2)
axes[0].set_title("Egitim Kayibi")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].bar(["Base", "LoRA"], [base_test, lora_test], color=["#3B8BD4", "#E24B4A"])
axes[1].set_title("Test MSE Karsilastirmasi")
axes[1].grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(r"c:\Users\cagan\Downloads\softitoprojelerim\proje-reposu\LLM\02-fine-tuning-lora\02-finans-siniflandirici\01_lora_comparison.png", dpi=120)
plt.close()

print(f"\n{'='*70}")
print(f"  TAMAMLANDI")
print(f"{'='*70}")
