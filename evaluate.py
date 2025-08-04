import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from cnn_model import AgeCNN

# Age label map
AGE_LIST = ['(0–2)', '(4–6)', '(8–12)', '(15–20)',
            '(25–32)', '(38–43)', '(48–53)', '(60–100)']

# Define transform (same as training)
transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor()
])

# Load trained model
model = AgeCNN()
model.load_state_dict(torch.load("age_cnn.pth"))
model.eval()

# Folder of test images
test_dir = "images"

for filename in os.listdir(test_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(test_dir, filename)
        image = Image.open(path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)  # add batch dim

        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            predicted_idx = torch.argmax(outputs, dim=1).item()
            predicted_label = AGE_LIST[predicted_idx]

        print(f"✅ {filename} => Predicted Age Group: {predicted_label}")
