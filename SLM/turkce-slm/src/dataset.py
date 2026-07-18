import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, token_ids, block_size=128):
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.token_ids) - self.block_size - 1)

    def __getitem__(self, idx):
        chunk = self.token_ids[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def create_dataloader(token_ids, block_size=128, batch_size=32, shuffle=True):
    dataset = TextDataset(token_ids, block_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    fake_ids = list(range(1000)) * 10
    dl = create_dataloader(fake_ids, block_size=64, batch_size=8)
    x, y = next(iter(dl))
    print(f"    Batch boyutu: {x.shape}")
    print(f"    Input: {x[0][:10]}...")
    print(f"    Target: {y[0][:10]}...")
