import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from tqdm import tqdm

from .dataset import SERDataset
from .models import CRNN
from . import config

def collate_fn(batch):
    """
    Pads/collates Mel spectrograms into (B, 1, NUM_MEL, T) tensor.
    """
    xs, ys = zip(*batch)
    xs = [torch.tensor(x) for x in xs]

    # Pad time dimension
    max_T = max(x.shape[1] for x in xs)
    padded = []
    for x in xs:
        pad = max_T - x.shape[1]
        padded.append(torch.nn.functional.pad(x, (0, pad)))

    X = torch.stack(padded)       # (B, NUM_MEL, T)
    X = X.unsqueeze(1)            # (B, 1, NUM_MEL, T)
    Y = torch.tensor(ys)
    return X.float(), Y.long()

def train_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    running_loss = 0

    for X, y in tqdm(loader, desc="Training"):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X.size(0)

    return running_loss / len(loader.dataset)

def validate(model, loader, device):
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    correct = 0
    total = 0
    val_loss = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            val_loss += loss.item() * X.size(0)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return val_loss / len(loader.dataset), correct / total

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    dataset = SERDataset()
    train_size = int(config.TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = CRNN().to(device)
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)

    for epoch in range(1, config.N_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.N_EPOCHS}")
        loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, device)
        print(f"Train Loss: {loss:.4f}  |  Val Loss: {val_loss:.4f}  |  Val Acc: {val_acc:.4f}")

        torch.save(model.state_dict(), f"ser_epoch{epoch}.pth")

if __name__ == "__main__":
    main()
