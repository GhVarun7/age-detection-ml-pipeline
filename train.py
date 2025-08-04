import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from cnn_model import AgeCNN
from torch_dataset import AgeLabeledDataset

# Config
BATCH_SIZE = 4
EPOCHS = 10
LR = 0.001

# Load dataset
dataset = AgeLabeledDataset("images", "labels.csv")
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Init model
model = AgeCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i+1) % 2 == 0:
            print(f"[Epoch {epoch+1}, Batch {i+1}] loss: {running_loss/2:.4f}")
            running_loss = 0.0

# Save model
torch.save(model.state_dict(), "age_cnn.pth")
print("✅ Training complete. Model saved as age_cnn.pth")
