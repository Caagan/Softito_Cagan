import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("  DPO EGITIMI — Direct Preference Optimization")
print("=" * 70)

print("\n[PROJE] Preference Modeli Egitimi (DPO)")
print("-" * 50)

class PreferenceDataset(Dataset):
    def __init__(self, n=1000, seq_len=20):
        self.data = []
        for _ in range(n):
            prompt = torch.randint(0, 100, (seq_len,))
            good_response = torch.randint(100, 200, (seq_len,))
            bad_response = torch.randint(200, 300, (seq_len,))
            self.data.append((prompt, good_response, bad_response))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class SimpleLM(nn.Module):
    def __init__(self, vocab_size=300, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        return self.fc(out[:, -1, :])

    def reward(self, x):
        logits = self.forward(x)
        return logits.sum(dim=-1)

def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    chosen_lograt = policy_chosen - ref_chosen
    rejected_lograt = policy_rejected - ref_rejected
    loss = -F.logsigmoid(beta * (chosen_lograt - rejected_lograt))
    return loss.mean()

dataset = PreferenceDataset(n=500)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

policy = SimpleLM()
ref_model = SimpleLM()
ref_model.load_state_dict(policy.state_dict())
ref_model.eval()

optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

print(f"    Parametre sayisi: {sum(p.numel() for p in policy.parameters()):,}")
print(f"    Veri boyutu: {len(dataset)}")

losses = []
accuracies = []

print(f"\n    Epoch  | Kayip  | Dogruluk")
print(f"    {'─'*38}")

for epoch in range(10):
    epoch_loss = 0
    correct = 0
    total = 0

    for prompt, good, bad in dataloader:
        with torch.no_grad():
            ref_chosen = ref_model.reward(good)
            ref_rejected = ref_model.reward(bad)

        policy_chosen = policy.reward(good)
        policy_rejected = policy.reward(bad)

        loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
        correct += (policy_chosen > policy_rejected).sum().item()
        total += len(prompt)

    avg_loss = epoch_loss / len(dataloader)
    acc = correct / total
    losses.append(avg_loss)
    accuracies.append(acc)

    if (epoch + 1) % 2 == 0:
        print(f"    {epoch+1:>5}  | {avg_loss:.4f} | {acc:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("DPO Egitimi — Sonuclar", fontsize=14)

axes[0].plot(losses, "b-o", linewidth=2)
axes[0].set_title("Kayip (Loss)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(alpha=0.3)

axes[1].plot(accuracies, "g-o", linewidth=2)
axes[1].set_title("Dogruluk (Chosen > Rejected)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(r"c:\Users\cagan\Downloads\softitoprojelerim\proje-reposu\LLM\06-rlhf-dpo\01_dpo_training.png", dpi=120)
plt.close()

print(f"\n{'='*70}")
print(f"  TAMAMLANDI — Final Dogruluk={accuracies[-1]:.4f}")
print(f"{'='*70}")
