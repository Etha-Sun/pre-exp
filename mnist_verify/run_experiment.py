"""
MNIST ring-feature experiment (CPU-only, macOS friendly).

This script will:
- Download/load MNIST
- Compute a binary ring (hole) feature per image
- Train (1) linear softmax and (2) 1-hidden-layer ReLU MLP on pixels+ring
- Encourage reliance on the ring feature via custom regularization
- Print explicit formulas and save weight visualizations to outputs/

Run:
  python mnist_verify/run_experiment.py
"""

import os
import math
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# -----------------------------
# Reproducibility and constants
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BATCH_SIZE = 256
INPUT_DIM = 28 * 28 + 1  # pixels + ring
NUM_CLASSES = 10
DATA_ROOT = str(Path.home() / '.torch' / 'datasets')
OUTPUT_DIR = Path(__file__).parent / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------
# Ring (hole) feature computation utils
# --------------------------------------
import collections


def compute_ring_feature(img: torch.Tensor, threshold: float = 0.5) -> int:
    """
    Given a grayscale MNIST image tensor of shape [1, 28, 28] in [0,1],
    returns 1 if there exists at least one background (0) connected component
    fully enclosed by foreground (1), else 0.
    Threshold rule: foreground if value > threshold.
    """
    assert img.ndim == 3 and img.shape[0] == 1, "Expected [1,H,W] image"
    h, w = img.shape[1], img.shape[2]
    x = (img[0] > threshold).cpu().numpy().astype(np.uint8)  # 1 for foreground strokes

    # Background mask (0 where foreground, 1 where background)
    bg = (x == 0).astype(np.uint8)

    # Flood-fill background from border to mark non-hole background
    visited = np.zeros_like(bg, dtype=np.uint8)
    dq = collections.deque()

    # Push border background pixels
    for i in range(h):
        if bg[i, 0] and not visited[i, 0]:
            visited[i, 0] = 1
            dq.append((i, 0))
        if bg[i, w - 1] and not visited[i, w - 1]:
            visited[i, w - 1] = 1
            dq.append((i, w - 1))
    for j in range(w):
        if bg[0, j] and not visited[0, j]:
            visited[0, j] = 1
            dq.append((0, j))
        if bg[h - 1, j] and not visited[h - 1, j]:
            visited[h - 1, j] = 1
            dq.append((h - 1, j))

    # 4-connected BFS
    OFFSETS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while dq:
        i, j = dq.popleft()
        for di, dj in OFFSETS:
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w and bg[ni, nj] and not visited[ni, nj]:
                visited[ni, nj] = 1
                dq.append((ni, nj))

    # Any background pixel not visited is a hole pixel
    holes = (bg == 1) & (visited == 0)
    return int(holes.any())


class RingMNIST(Dataset):
    def __init__(self, base: datasets.MNIST, threshold: float = 0.5):
        self.base = base
        self.threshold = threshold

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        img, label = self.base[idx]  # img: [1,28,28] float in [0,1]
        ring = compute_ring_feature(img, threshold=self.threshold)
        pixels = img.view(-1)
        feat = torch.cat([pixels, torch.tensor([float(ring)], dtype=pixels.dtype)])
        return feat, label, torch.tensor(ring, dtype=torch.float32)


# -----------------------------
# Data loading (CPU-only safe)
# -----------------------------
def get_ring_loaders(batch_size: int = BATCH_SIZE, threshold: float = 0.5):
    tf = transforms.Compose([transforms.ToTensor()])
    base_train = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=tf)
    base_test = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=tf)

    ring_train = RingMNIST(base_train, threshold=threshold)
    ring_test = RingMNIST(base_test, threshold=threshold)

    def collate(batch):
        feats, labels, rings = zip(*batch)
        return torch.stack(feats), torch.tensor(labels, dtype=torch.long), torch.stack(rings)

    # num_workers=0 for notebook/mac stability; pin_memory=False (CPU-only)
    train_loader = DataLoader(
        ring_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, collate_fn=collate
    )
    test_loader = DataLoader(
        ring_test, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=collate
    )
    return train_loader, test_loader


# -----------------------------
# Models
# -----------------------------
class LinearSoftmax(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.W = nn.Linear(in_dim, num_classes, bias=True)

    def forward(self, x):
        return self.W(x)


class OneHiddenMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h)


# ----------------------------------------------
# Regularization to encourage ring-feature usage
# ----------------------------------------------
def ring_weight_penalty_linear(model: LinearSoftmax, alpha: float = 1.0) -> torch.Tensor:
    # last input is the ring feature
    W = model.W.weight  # [C, D]
    ring_w = W[:, -1]   # [C]
    return -alpha * torch.mean(ring_w.abs())  # negative encourages larger magnitude


def ring_weight_penalty_mlp(model: OneHiddenMLP, beta: float = 1.0) -> torch.Tensor:
    # Encourage large absolute weights from ring input into hidden units
    W1 = model.fc1.weight  # [H, D]
    ring_w1 = W1[:, -1]    # [H]
    return -beta * torch.mean(ring_w1.abs())


# -----------------------------
# Training loop (CPU)
# -----------------------------
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 3,
    lr: float = 1e-2,
    ring_reg: float = 0.0,
    model_type: str = 'linear',
):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for xb, yb, rb in train_loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            if ring_reg > 0:
                if model_type == 'linear':
                    loss = loss + ring_reg * ring_weight_penalty_linear(model)
                else:
                    loss = loss + ring_reg * ring_weight_penalty_mlp(model)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss) * xb.size(0)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total += xb.size(0)
        train_loss = total_loss / total
        train_acc = total_correct / total

        # Eval
        model.eval()
        correct = 0
        count = 0
        with torch.no_grad():
            for xb, yb, rb in test_loader:
                logits = model(xb)
                correct += (logits.argmax(dim=1) == yb).sum().item()
                count += xb.size(0)
        test_acc = correct / count
        print({'epoch': epoch, 'train_loss': round(train_loss, 4), 'train_acc': round(train_acc, 4), 'test_acc': round(test_acc, 4)})
    return model


# -----------------------------
# Reporting helpers
# -----------------------------
def save_linear_reports(W_lin: np.ndarray, b_lin: np.ndarray, output_dir: Path):
    ring_weights = W_lin[:, -1]
    # Save ring weights bar
    plt.figure(figsize=(6, 3))
    plt.bar(np.arange(NUM_CLASSES), ring_weights)
    plt.title('Linear: ring weight per class')
    plt.xlabel('class')
    plt.ylabel('ring weight')
    plt.tight_layout()
    plt.savefig(output_dir / 'linear_ring_weight_per_class.png', dpi=150)
    plt.close()

    # Save pixel weights heatmaps
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for c, ax in enumerate(axes.flat):
        ax.imshow(W_lin[c, :-1].reshape(28, 28), cmap='coolwarm')
        ax.set_title(f'class {c}')
        ax.axis('off')
    plt.suptitle('Linear: pixel weights per class')
    plt.tight_layout()
    plt.savefig(output_dir / 'linear_pixel_weights.png', dpi=150)
    plt.close()


def save_mlp_reports(w1: torch.Tensor, output_dir: Path):
    ring_in_w = w1[:, -1].abs().numpy()
    plt.figure(figsize=(6, 3))
    plt.hist(ring_in_w, bins=20)
    plt.title('MLP: |ring -> hidden| weight distribution')
    plt.tight_layout()
    plt.savefig(output_dir / 'mlp_ring_to_hidden_weight_hist.png', dpi=150)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    print({'data_root': DATA_ROOT})
    train_loader, test_loader = get_ring_loaders()

    # Train linear model
    linear = LinearSoftmax(INPUT_DIM, NUM_CLASSES)
    linear = train_model(linear, train_loader, test_loader, epochs=5, lr=1e-3, ring_reg=1e-3, model_type='linear')

    # Extract explicit formula: logits = W x + b
    W_lin = linear.W.weight.detach().cpu().numpy()   # [10, 785]
    b_lin = linear.W.bias.detach().cpu().numpy()     # [10]
    print({'linear_ring_weight_per_class': np.round(W_lin[:, -1], 4).tolist()})
    save_linear_reports(W_lin, b_lin, OUTPUT_DIR)

    # Train 1-hidden-layer MLP
    mlp = OneHiddenMLP(INPUT_DIM, hidden_dim=64, num_classes=NUM_CLASSES)
    mlp = train_model(mlp, train_loader, test_loader, epochs=5, lr=1e-3, ring_reg=1e-3, model_type='mlp')

    with torch.no_grad():
        w1 = mlp.fc1.weight.detach().cpu()
        w2 = mlp.fc2.weight.detach().cpu()
    print({'mlp_mean_abs_ring_to_hidden': float(w1[:, -1].abs().mean())})
    save_mlp_reports(w1, OUTPUT_DIR)

    # Print explicit forms summary
    print('Linear softmax explicit form: logit_c = b_c + sum_i W[c,i] * x[i] + W[c,ring] * ring')
    print('MLP explicit form: logits = W2 * ReLU(W1 * x_ext) + b2, where x_ext = [pixels; ring]')
    print({'outputs_saved_to': str(OUTPUT_DIR.resolve())})


if __name__ == '__main__':
    main()


