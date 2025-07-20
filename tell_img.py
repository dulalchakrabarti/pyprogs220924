import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# %%
# Load pre-trained model
model = models.efficientnet_b0(weights='DEFAULT')
model.eval()
# %%
# Define crop classes (example - modify with your specific crops)
crop_classes = [
    'Wheat', 'Corn/Maize', 'Rice', 'Soybean', 'Barley',
    'Cotton', 'Sugarcane', 'Tomato', 'Potato', 'Grapes'
]

# Modify model for our crop classes
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(crop_classes))
# %%
# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# %%
def classify_crop_image(img_source, model, top_k=3):
    """Classify a crop image from file path or URL"""
    # Load image
    if img_source.startswith('http'):
        response = requests.get(img_source)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(img_source)
    
    # Preprocess and predict
    img_tensor = preprocess(img)
    batch = img_tensor.unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(batch)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
    # Get top predictions
    top_probs, top_indices = torch.topk(probs, top_k)
    results = [(crop_classes[i], p.item()) for i, p in zip(top_indices, top_probs)]
    
    # Display results
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
    for i, (crop, prob) in enumerate(results):
        print(f"{i+1}. {crop}: {prob*100:.2f}%")
    
    return results
# %%
# Example usage with sample image
#sample_url = "https://farm5.staticflickr.com/4072/4289549684_8e587245a3_z.jpg"
#classify_crop_image(sample_url, model)

# %%
# Upload your own image
#from google.colab import files

#uploaded = files.upload()
filenames=["/media/dc/DC-28122024/wheatcam3/Wheat/image_0463.jpg"]
for filename in filenames:
    print(f"\nClassifying: {filename}")
    classify_crop_image(filename, model)


