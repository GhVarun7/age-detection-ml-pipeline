import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class AgeLabeledDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.labels_df = pd.read_csv(label_file)
        self.transform = transform or transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = int(row["label"])
        return image, label
