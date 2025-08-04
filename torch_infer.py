import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
from torchvision.transforms.functional import to_pil_image
from torch_dataset import AgeDataset
from utils import load_age_model, predict_age

# Load existing Caffe model
age_net = load_age_model("model/deploy_age.prototxt", "model/age_net.caffemodel")

# Load dataset using PyTorch
dataset = AgeDataset("images")
loader = DataLoader(dataset, batch_size=1, shuffle=False)

for img_tensor, filename in loader:
    # Convert tensor to NumPy
    img_array = img_tensor.squeeze(0).permute(1, 2, 0).numpy()
    img_array = (img_array * 255).astype(np.uint8)
    face_img = cv2.resize(img_array, (227, 227))

    # Run prediction
    age = predict_age(face_img, age_net)

    print(f"✅ {filename[0]} => Age: {age}")
