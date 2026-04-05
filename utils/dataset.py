import os
import cv2
from torch.utils.data import Dataset

class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        classes = sorted(os.listdir(root_dir))

        for label, cls in enumerate(classes):
            cls_path = os.path.join(root_dir, cls)
            for img in os.listdir(cls_path):
                self.image_paths.append(os.path.join(cls_path, img))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        return img, label