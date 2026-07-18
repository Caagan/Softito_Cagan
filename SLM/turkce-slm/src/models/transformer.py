import torch
import torch.nn as nn
import math

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=4,
                 max_len=512, ff_dim=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        self.embed_dim = embed_dim

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(0, T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.embedding(x) + self.pos_embed(positions)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        logits = self.head(h)
        return logits

if __name__ == "__main__":
    model = MiniGPT(vocab_size=1000, embed_dim=128, num_heads=4, num_layers=4)
    params = sum(p.numel() for p in model.parameters())
    print(f"    Model parametre sayisi: {params:,}")

    dummy_input = torch.randint(0, 1000, (2, 64))
    output = model(dummy_input)
    print(f"    Input boyutu: {dummy_input.shape}")
    print(f"    Output boyutu: {output.shape}")
