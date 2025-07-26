import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from load_resnet import load_resnet50_v2
from wilds.common.grouper import CombinatorialGrouper
from tqdm import tqdm

# Hyperparameters
BATCH_SIZE = 64
LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 500
LOG_DIR = "runs/waterbirds_finetune_wilds_tqdm"
CHECKPOINT_DIR = "saved_models"
BEST_VAL_MODEL = "best_val_resnet50_waterbirds.pth"
BEST_TRAIN_MODEL = "best_train_resnet50_waterbirds.pth"
DEVICE = "cuda:0"
SEED = 0  # for reproducibility

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Set seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# There are 4 groups: (label × background) combinations
NUM_GROUPS = 4

def finetune():
    # Device setup
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Seed: {SEED}\n")

    # Load pretrained ResNet-50 v2 and replace head
    model, weights = load_resnet50_v2(device)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    # Load dataset via WILDS
    dataset = get_dataset(dataset="waterbirds", download=True)
    grouper = CombinatorialGrouper(dataset, ['y', 'background'])

    train_data = dataset.get_subset("train", transform=weights.transforms())
    val_data   = dataset.get_subset("val",   transform=weights.transforms())
    test_data  = dataset.get_subset("test",  transform=weights.transforms())

    # Create data loaders
    train_loader = get_train_loader("standard", train_data, batch_size=BATCH_SIZE)
    val_loader   = get_eval_loader("standard", val_data,   batch_size=BATCH_SIZE)
    test_loader  = get_eval_loader("standard", test_data,  batch_size=BATCH_SIZE)

    # Loss, optimizer, TensorBoard
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    writer = SummaryWriter(log_dir=LOG_DIR)

    # Tracking best metrics
    best_val_acc = 0.0
    best_train_acc = 0.0

    print("Starting fine-tuning...")
    print("-" * 60)
    for epoch in range(1, EPOCHS + 1):
        # ----- Training -----
        model.train()
        train_loss = 0.0
        train_correct = 0
        total = 0
        group_correct = [0] * NUM_GROUPS
        group_total   = [0] * NUM_GROUPS
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        for x, y, metadata in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            preds = logits.argmax(1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            train_loss += loss.item() * bs
            train_correct += (preds == y).sum().item()
            total += bs

            # Group-wise
            group_ids = grouper.metadata_to_group(metadata)
            for i in range(bs):
                gid = int(group_ids[i])
                group_total[gid] += 1
                if preds[i] == y[i]:
                    group_correct[gid] += 1

            loop.set_postfix(train_loss=f"{train_loss/total:.4f}", train_acc=f"{train_correct/total:.4f}")

        train_loss /= total
        train_acc = train_correct / total

        train_group_acc = [group_correct[i] / group_total[i] if group_total[i] > 0 else 0.0
                         for i in range(NUM_GROUPS)]

        # ----- Validation -----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total = 0
        group_correct = [0] * NUM_GROUPS
        group_total   = [0] * NUM_GROUPS
        with torch.no_grad():
            for x, y, metadata in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                bs = x.size(0)
                val_loss += loss.item() * bs
                preds = logits.argmax(1)
                val_correct += (preds == y).sum().item()
                total += bs

                # Group-wise
                group_ids = grouper.metadata_to_group(metadata)
                for i in range(bs):
                    gid = int(group_ids[i])
                    group_total[gid] += 1
                    if preds[i] == y[i]:
                        group_correct[gid] += 1

        val_loss /= total
        val_acc = val_correct / total
        val_group_acc = [group_correct[i] / group_total[i] if group_total[i] > 0 else 0.0
                         for i in range(NUM_GROUPS)]

        # ----- Test -----
        test_correct = 0
        total = 0
        test_group_correct = [0] * NUM_GROUPS
        test_group_total   = [0] * NUM_GROUPS
        with torch.no_grad():
            for x, y, metadata in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(1)
                bs = x.size(0)
                test_correct += (preds == y).sum().item()
                total += bs

                # Group-wise
                group_ids = grouper.metadata_to_group(metadata)
                for i in range(bs):
                    gid = int(group_ids[i])
                    test_group_total[gid] += 1
                    if preds[i] == y[i]:
                        test_group_correct[gid] += 1

        test_acc = test_correct / total
        test_group_acc = [test_group_correct[i] / test_group_total[i] if test_group_total[i] > 0 else 0.0
                          for i in range(NUM_GROUPS)]

        # ----- Logging -----
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/train",  train_acc,  epoch)
        writer.add_scalar("Loss/val",   val_loss,   epoch)
        writer.add_scalar("Acc/val",    val_acc,    epoch)
        writer.add_scalar("Acc/test",   test_acc,   epoch)
        for i in range(NUM_GROUPS):
            writer.add_scalar(f"Acc/val_group_{i}", val_group_acc[i], epoch)
            writer.add_scalar(f"Acc/test_group_{i}", test_group_acc[i], epoch)

        # ----- Pretty Printing -----
        print(f"Epoch {epoch:02d}/{EPOCHS}")
        print()
        print(f"Train   | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} (best so far: {best_train_acc:.4f})")
        print(f"Val     | Loss: {val_loss:.4f} | Acc: {val_acc:.4f} (best so far: {best_val_acc:.4f})")
        print(f"Test    | Acc: {test_acc:.4f}")
        print()
        print(f"Train Group accs : {[f'{acc:.3f}' for acc in train_group_acc]} ")
        print(f"Val Group accs   : {[f'{acc:.3f}' for acc in val_group_acc]} ")
        print(f"Test Group accs  : {[f'{acc:.3f}' for acc in test_group_acc]} ")
        print("-" * 60)

        # ----- Checkpoints -----
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, BEST_TRAIN_MODEL))
            print(f"✓ New best TRAIN acc ({best_train_acc:.4f}) saved to {BEST_TRAIN_MODEL}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, BEST_VAL_MODEL))
            print(f"✓ New best VAL   acc ({best_val_acc:.4f}) saved to {BEST_VAL_MODEL}")

        if train_acc == 1.0:
            print("\n\n")
            print("Reached perfect training accuracy, stopping.")
            break

        print("\n\n")


    writer.close()

if __name__ == "__main__":
    finetune()
