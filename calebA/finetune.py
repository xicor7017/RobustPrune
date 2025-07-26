import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CelebA
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

# Hyperparameters
BATCH_SIZE       = 256
LR               = 1e-4
WEIGHT_DECAY     = 1e-4
EPOCHS           = 500
LOG_DIR          = "runs/celeba_finetune"
CHECKPOINT_DIR   = "saved_models"
BEST_VAL_MODEL   = "best_val_resnet50_celeba.pth"
BEST_TRAIN_MODEL = "best_train_resnet50_celeba.pth"
DEVICE           = "cuda"            # use CUDA devices
SEED             = 0                 # reproducibility
N_GPUS           = 2                 # number of GPUs to use

# Set up
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark   = False

# Custom wrapper to pick out the 'Male' attribute (binary gender)
class CelebAGenderDataset(Dataset):
    def __init__(self, root, split, transform=None, download=True):
        self.base = CelebA(root=root,
                           split=split,
                           target_type='attr',
                           transform=transform,
                           download=download)
        self.gender_idx = self.base.attr_names.index('Male')
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        img, attrs = self.base[idx]
        label = attrs[self.gender_idx].long()
        return img, label

def finetune():
    # Device setup
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for multi‑GPU training")
    device = torch.device(DEVICE)
    print(f"Using device: {device}, GPUs: {min(N_GPUS, torch.cuda.device_count())} | Seed: {SEED}\n")

    # Load pretrained ResNet-50 v2
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).to(device)
    model.fc = nn.Linear(model.fc.in_features, 2).to(device)

    # Wrap model for data parallelism
    available_gpus = torch.cuda.device_count()
    if available_gpus > 1 and N_GPUS > 1:
        use_gpus = list(range(min(N_GPUS, available_gpus)))
        model = nn.DataParallel(model, device_ids=use_gpus)
        print(f"→ Model wrapped in DataParallel on GPUs: {use_gpus}")

    # ImageNet‑style preprocessing
    transform = weights.transforms()

    # Datasets & loaders
    train_ds = CelebAGenderDataset(root="data/celeba", split="train",
                                   transform=transform, download=True)
    val_ds   = CelebAGenderDataset(root="data/celeba", split="valid",
                                   transform=transform, download=False)
    test_ds  = CelebAGenderDataset(root="data/celeba", split="test",
                                   transform=transform, download=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Criterion, optimizer, TensorBoard
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    writer    = SummaryWriter(log_dir=LOG_DIR)

    best_val_acc   = 0.0
    best_train_acc = 0.0

    print("Starting fine-tuning...\n" + "-"*60)
    for epoch in range(1, EPOCHS+1):
        # ----- Train -----
        model.train()
        running_loss = 0.0
        correct = 0
        total   = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            bs = imgs.size(0)
            running_loss += loss.item() * bs
            preds = logits.argmax(1)
            correct     += (preds == labels).sum().item()
            total       += bs
            pbar.set_postfix(Running_train_loss=f"{running_loss/total:.4f}",
                             Running_train_acc =f"{correct/total:.4f}")

        train_loss = running_loss/total
        train_acc  = correct/total

        # ----- Validate -----
        model.eval()
        val_loss = 0.0
        correct  = 0
        total    = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss   = criterion(logits, labels)
                bs     = imgs.size(0)
                val_loss += loss.item() * bs
                preds    = logits.argmax(1)
                correct  += (preds == labels).sum().item()
                total    += bs
        val_loss /= total
        val_acc   = correct/total

        # ----- Test -----
        test_acc = 0.0
        correct  = 0
        total    = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                correct += (preds == labels).sum().item()
                total   += imgs.size(0)
        test_acc = correct/total

        # Log to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/train",  train_acc,   epoch)
        writer.add_scalar("Loss/val",   val_loss,    epoch)
        writer.add_scalar("Acc/val",    val_acc,     epoch)
        writer.add_scalar("Acc/test",   test_acc,    epoch)

        # Pretty print
        print(f"\nEpoch {epoch:02d}/{EPOCHS}")
        print(f"▶ Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} (best {best_train_acc:.4f})")
        print(f"▶ Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.4f} (best {best_val_acc:.4f})")
        print(f"▶ Test  | Acc:  {test_acc:.4f}\n" + "-"*60)

        # Checkpointing
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, BEST_TRAIN_MODEL))
            print(f"✓ New best TRAIN acc: {best_train_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, BEST_VAL_MODEL))
            print(f"✓ New best VAL   acc: {best_val_acc:.4f}")

        if train_acc == 1.0:
            print("Perfect training accuracy reached; stopping early.")
            break

    writer.close()

if __name__ == "__main__":
    finetune()
