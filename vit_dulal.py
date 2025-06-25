'''
wheat_yield_project/
├── images/
│   ├── plot_001.jpg
│   ├── plot_002.jpg
├── labels.csv  # plot_id,yield
├── model.py
├── dataset.py
└── train.py
'''
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16

class WheatYieldViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = vit_b_16(pretrained=True)
        self.vit.heads = nn.Linear(self.vit.heads.in_features, 1)  # Regression

    def forward(self, x):
        return self.vit(x)


import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from torchvision import transforms

class WheatDataset(Dataset):
    def __init__(self, csv_file, image_dir):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, f"{row['plot_id']}.jpg")
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(row['yield'], dtype=torch.float32)
        return image, label


import torch
from torch.utils.data import DataLoader
from model import WheatYieldViT
from dataset import WheatDataset
import torch.nn as nn
import torch.optim as optim

dataset = WheatDataset("labels.csv", "images/")
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = WheatYieldViT()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(30):
    model.train()
    total_loss = 0
    for imgs, targets in loader:
        preds = model(imgs).squeeze()
        loss = criterion(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")


import torch
import pandas as pd
import matplotlib.pyplot as plt
from model import WheatYieldViT
from dataset import WheatDataset
from torch.utils.data import DataLoader
import torch.nn as nn

# Load model
model = WheatYieldViT()
model.load_state_dict(torch.load("wheat_yield_vit.pth"))  # Load your trained model
model.eval()

# Load dataset
test_dataset = WheatDataset("labels.csv", "images/")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Evaluate
predictions = []
actuals = []

with torch.no_grad():
    for img, label in test_loader:
        output = model(img)
        predictions.append(output.item())
        actuals.append(label.item())

# Convert to DataFrame
df_results = pd.DataFrame({
    "Actual Yield": actuals,
    "Predicted Yield": predictions
})

# Save results to CSV
df_results.to_csv("yield_predictions.csv", index=False)

# Plot Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(actuals, predictions, color="cornflowerblue", edgecolor="black")
plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], "r--")
plt.xlabel("Actual Yield (tons/ha)")
plt.ylabel("Predicted Yield (tons/ha)")
plt.title("Wheat Yield Prediction: Actual vs. Predicted")
plt.grid(True)
plt.tight_layout()
plt.savefig("yield_scatter.png")
plt.show()

# Optional: Compute RMSE
rmse = nn.MSELoss()(torch.tensor(predictions), torch.tensor(actuals)).sqrt()
print(f"RMSE: {rmse:.2f} tons/ha")
