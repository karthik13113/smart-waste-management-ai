import os
import torch
from dataset import ImageFolderDataset
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- MANUAL ARGUMENTS FOR SPYDER ----------
class Args:
    data_dir = "data"                     # dataset folder
    model = "outputs/best_model.pth"      # trained model path
    out = "outputs"                       # output folder

args = Args()
# -------------------------------------------------

def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    import torchvision.models as models
    model = models.mobilenet_v2(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(num_ftrs, len(ckpt['classes']))
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()
    return model, ckpt['classes']

@torch.no_grad()
def evaluate(model, loader, device):
    preds, trues = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds.extend(outputs.argmax(1).cpu().numpy().tolist())
        trues.extend(labels.numpy().tolist())
    return preds, trues

def plot_confusion(trues, preds, classes, out_path):
    cm = confusion_matrix(trues, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Loading model:", args.model)
    model, classes = load_model(args.model, device)

    print("Loading test dataset...")
    test_ds = ImageFolderDataset(os.path.join(args.data_dir, 'test'))
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    preds, trues = evaluate(model, test_loader, device)

    report = classification_report(trues, preds, target_names=test_ds.classes, digits=4)
    print(report)

    with open(os.path.join(args.out, 'classification_report.txt'), 'w') as f:
        f.write(report)

    plot_confusion(trues, preds, test_ds.classes, os.path.join(args.out, 'confusion_matrix.png'))
    print("\n✔ Saved report and confusion matrix to", args.out)

if __name__ == '__main__':
    main()
