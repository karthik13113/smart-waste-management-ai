import argparse, os, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from dataset import ImageFolderDataset
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    preds, trues = [], []
    for imgs, labels in tqdm(loader, desc='train', leave=False):
        imgs = imgs.to(device); labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds.extend(outputs.argmax(1).cpu().numpy().tolist())
        trues.extend(labels.cpu().numpy().tolist())
    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1 = f1_score(trues, preds, average='macro')
    return epoch_loss, epoch_f1

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds, trues = [], []
    for imgs, labels in tqdm(loader, desc='val', leave=False):
        imgs = imgs.to(device); labels = labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * imgs.size(0)
        preds.extend(outputs.argmax(1).cpu().numpy().tolist())
        trues.extend(labels.cpu().numpy().tolist())
    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1 = f1_score(trues, preds, average='macro')
    return epoch_loss, epoch_f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--out_dir', type=str, default='outputs')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)

    # Datasets
    train_ds = ImageFolderDataset(os.path.join(args.data_dir, 'train'))
    val_ds = ImageFolderDataset(os.path.join(args.data_dir, 'val'))
    classes = train_ds.classes
    print('Classes:', classes)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model - MobileNetV2 backbone
    model = models.mobilenet_v2(pretrained=True)
    num_ftrs = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(num_ftrs, len(classes))
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_f1 = 0.0
    best_path = None

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1 = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        print(f'Epoch {epoch}/{args.epochs} - time: {elapsed:.1f}s')
        print(f'  train loss: {train_loss:.4f} f1: {train_f1:.4f}')
        print(f'  val   loss: {val_loss:.4f} f1: {val_f1:.4f}')
        # save best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_path = os.path.join(args.out_dir, 'best_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': classes
            }, best_path)
            print('  Saved best model to', best_path)
    print('Training complete. Best val f1:', best_val_f1)
    print('Best model path:', best_path)

if __name__ == '__main__':
    main()
