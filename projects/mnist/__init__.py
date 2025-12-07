import os
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import wandb
import json
from datetime import datetime

@dataclass
class Config:
    batch_size: int = 128
    lr: float = 1e-3
    num_epochs: int = 3
    weight_decay: float = 0.0
    num_workers: int = 2

    model_hidden: int = 256
    seed: int = 42
    log_interval: int = 100            # steps

    use_cuda: bool = True

class MLP(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10),
        )

    def forward(self, x):
        return self.net(x)

def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataloaders(cfg: Config):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_ds = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_ds = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def train_one_epoch(
    model, device, train_loader, optimizer, criterion, epoch: int, cfg: Config
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        _, preds = outputs.max(1)
        total += target.size(0)
        correct += preds.eq(target).sum().item()

        if (batch_idx + 1) % cfg.log_interval == 0:
            step = epoch * len(train_loader) + batch_idx
            wandb.log(
                {
                    "train/batch_loss": loss.item(),
                    "train/batch_acc": preds.eq(target).float().mean().item(),
                    "train/step": step,
                }
            )

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    wandb.log(
        {"train/epoch_loss": epoch_loss, "train/epoch_acc": epoch_acc, "epoch": epoch}
    )

    print(
        f"Epoch {epoch}: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}"
    )


@torch.no_grad()
def evaluate(model, device, test_loader, criterion, epoch: int):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        loss = criterion(outputs, target)

        running_loss += loss.item() * data.size(0)
        _, preds = outputs.max(1)
        total += target.size(0)
        correct += preds.eq(target).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    wandb.log(
        {"val/loss": epoch_loss, "val/acc": epoch_acc, "epoch": epoch}
    )
    print(f"Epoch {epoch}: Val Loss: {epoch_loss:.4f}, Val Acc: {epoch_acc:.4f}")


def main(run, config):

    cfg = Config()
    # populate config with values from other config
    for k, v in config.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    run.config.update(asdict(cfg)) # need better config handling lol

    set_seed(cfg.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.use_cuda else "cpu"
    )
    print("Using device:", device)

    train_loader, test_loader = get_dataloaders(cfg)

    model = MLP(hidden=cfg.model_hidden).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # Track gradients & model topology
    wandb.watch(model, log="all", log_freq=cfg.log_interval)

    # Training loop
    for epoch in range(1, cfg.num_epochs + 1):
        train_one_epoch(model, device, train_loader, optimizer, criterion, epoch, cfg)
        evaluate(model, device, test_loader, criterion, epoch)

    os.makedirs("./out", exist_ok=True)

    # Save model weights
    model_path = os.path.join("./out", "mnist_mlp.pt")
    torch.save(model.state_dict(), model_path)

    run.save(model_path, policy="now")


    print("Finished training; model & config uploaded to W&B.")
    run.finish()
