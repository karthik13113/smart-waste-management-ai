import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class ImageFolderDataset(Dataset):
    """
    Simple image-folder dataset loader.
    Expects:
    root/
      class1/
        img1.jpg
        ...
      class2/
        ...
    """
    def __init__(self, root, transform=None):
        self.root = root
        self.classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}
        self.samples = []
        for c in self.classes:
            cdir = os.path.join(root, c)
            for fname in os.listdir(cdir):
                if fname.lower().endswith(('.jpg','.jpeg','.png','bmp')):
                    self.samples.append((os.path.join(cdir, fname), self.class_to_idx[c]))
        self.transform = transform or T.Compose([
            T.Resize((224,224)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225])
        ])
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, label
