import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.classifier import EfficientNetClassifier
from utils.dataset import BrainTumorDataset
from utils.augmentations import *
from config import *
from tqdm import tqdm
from evaluation import evaluate_model

def main():
    train_data = BrainTumorDataset("data/train", get_train_transform())
    val_data = BrainTumorDataset("data/val", get_val_transform())

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    model = EfficientNetClassifier(NUM_CLASSES).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))

    best_acc = 0

    for epoch in range(10):
        
        model.train()
        correct, total = 0, 0

        for imgs, labels in tqdm(train_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # Validation
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch+1}: Train {train_acc:.4f}, Val {val_acc:.4f}")

        scheduler.step(1 - val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Saved Best Model")

    print("Final Best Accuracy:", best_acc)

    print("\n🔍 Evaluating Best Model on Validation Set...")

    model.load_state_dict(torch.load("best_model.pth"))

    evaluate_model(model, val_loader, DEVICE)

if __name__ == "__main__":
    main()

