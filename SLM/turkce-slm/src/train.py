import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import create_dataloader
from src.models.transformer import MiniGPT
import matplotlib.pyplot as plt

def train(model, train_loader, val_loader, epochs=20, lr=3e-4, device="cpu"):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(scheduler.get_last_lr()[0])

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:>3}: train={train_loss:.4f}, val={val_loss:.4f}")

    return history

if __name__ == "__main__":
    vocab_size = 2000
    block_size = 64
    batch_size = 16
    epochs = 20

    print("=" * 70)
    print("  SLM EGITIM — MiniGPT Turkce")
    print("=" * 70)

    import random
    token_ids = [random.randint(0, vocab_size-1) for _ in range(10000)]
    train_loader = create_dataloader(token_ids[:8000], block_size, batch_size)
    val_loader = create_dataloader(token_ids[8000:], block_size, batch_size, shuffle=False)

    model = MiniGPT(vocab_size=vocab_size, embed_dim=128, num_heads=4, num_layers=4, max_len=block_size)
    params = sum(p.numel() for p in model.parameters())
    print(f"    Model: {params:,} parametre")

    history = train(model, train_loader, val_loader, epochs=epochs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("SLM Egitim Sonuclari", fontsize=14)

    axes[0].plot(history["train_loss"], label="Train", linewidth=2)
    axes[0].plot(history["val_loss"], label="Validation", linewidth=2)
    axes[0].set_title("Kayip (Loss)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["lr"], "g-", linewidth=2)
    axes[1].set_title("Ogrenme Hizi (LR)")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(r"c:\Users\cagan\Downloads\softitoprojelerim\proje-reposu\SLM\turkce-slm\figures\egitim_sonuclari.png", dpi=120)
    plt.close()

    print(f"\n    Kayip: {history['train_loss'][-1]:.4f}")
    print(f"    Dogruluk (1-loss): {1 - history['val_loss'][-1]:.4f}")
