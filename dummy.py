
import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
    Normalize,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomRotation
)

from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer
from datasets import Dataset, Image

# --- 2. Load Dataset into Hugging Face Dataset format ---
def load_crop_dataset(base_dir='/media/dc/DC-28122024/data/crop_img224x224/'):
    image_paths = []
    labels = []
    class_names = sorted(os.listdir(base_dir))
    label_to_id = {class_name: i for i, class_name in enumerate(class_names)}
    id_to_label = {i: class_name for i, class_name in enumerate(class_names)}

    print(f"Detected classes: {class_names}")

    for class_name in class_names:
        class_dir = os.path.join(base_dir, class_name)
        if os.path.isdir(class_dir):
            for img_path in glob.glob(os.path.join(class_dir, "*")):
                image_paths.append(img_path)
                labels.append(label_to_id[class_name])

    # Create a dictionary for the dataset
    data = {"image": image_paths, "labels": labels}
   # Convert to Hugging Face Dataset
    # Use .cast(Image()) to tell the Dataset that 'image' column contains image paths
    # so it can load them as PIL Image objects.
    dataset = Dataset.from_dict(data).cast_column("image", Image())
    
    # Split into train and test
    train_test_split_ratio = 0.2
    train_val_split = dataset.train_test_split(test_size=train_test_split_ratio, seed=42)
    train_ds = train_val_split["train"]
    val_ds = train_val_split["test"] # Using 'test' as validation for simplicity

    return train_ds, val_ds, id_to_label, label_to_id, class_names

train_dataset, val_dataset, id_to_label, label_to_id, class_names = load_crop_dataset()

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Number of classes: {len(class_names)}")
print(f"Class mapping: {id_to_label}")
#'''
# Display a sample image
sample = train_dataset[0]
print(f"Sample image label: {id_to_label[sample['labels']]}")
plt.imshow(sample['image'])
plt.title(f"Label: {id_to_label[sample['labels']]}")
plt.axis('off')
plt.show()
# --- 3. Preprocessing ---
model_checkpoint = "google/vit-base-patch16-224-in21k"
image_processor = ViTImageProcessor.from_pretrained(model_checkpoint)

# Define data transformations for training and validation
# Augmentations for training:
train_transforms = Compose([
    RandomResizedCrop(image_processor.size["height"]),
    RandomHorizontalFlip(),
    RandomRotation(15),
    ToTensor(),
    Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
])

# No augmentations for validation, just resize and normalize
val_transforms = Compose([
    Resize((image_processor.size["height"], image_processor.size["width"])),
    ToTensor(),
    Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
])

def preprocess_train(example_batch):
    # 'image' column in Hugging Face Dataset holds PIL Images
    images = [img.convert("RGB") for img in example_batch['image']]
    # Apply torchvision transforms
    example_batch["pixel_values"] = [train_transforms(image) for image in images]
    return example_batch

def preprocess_val(example_batch):
    images = [img.convert("RGB") for img in example_batch['image']]
    example_batch["pixel_values"] = [val_transforms(image) for image in images]
    return example_batch

# Apply preprocessing
train_dataset.set_transform(preprocess_train)
val_dataset.set_transform(preprocess_val)

# Data collator to batch inputs
def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }
# --- 4. Load Model and Define Metrics ---
num_labels = len(class_names)

model = ViTForImageClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    id2label=id_to_label,
    label2id=label_to_id,
    ignore_mismatched_sizes=True # Ignore if the pre-trained head size doesn't match num_labels
)

# Define evaluation metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted') # 'weighted' for imbalanced classes
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    
    # Optionally, print confusion matrix for validation (be careful with large datasets)
    # cm = confusion_matrix(labels, predictions)
    # print("\nConfusion Matrix:\n", cm)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }
# --- 5. Configure Training Arguments and Trainer ---
training_args = TrainingArguments(
    output_dir="./vit-crop-classifier",            # output directory
    per_device_train_batch_size=16,                # batch size per device during training
    per_device_eval_batch_size=16,                 # batch size for evaluation
    eval_strategy="epoch",                   # evaluate each epoch
    num_train_epochs=5,                            # total number of training epochs
    fp16=torch.cuda.is_available(),                # use mixed precision (half-precision floats) if CUDA is available
    save_strategy="epoch",                                # save checkpoint every 100 steps
    eval_steps=100,                                # evaluate every 100 steps
    logging_steps=100,                             # log every 100 steps
    learning_rate=2e-5,                            # learning rate
    save_total_limit=2,                            # limit the number of saved checkpoints
    remove_unused_columns=False,                   # necessary for custom datasets
    load_best_model_at_end=True,                   # load the best model when training ends
    metric_for_best_model="f1",                    # metric to use for saving the best model
    report_to="none",                              # don't report to any specific platform (e.g., "tensorboard")
    seed=42,                                       # for reproducibility
    dataloader_num_workers=os.cpu_count() // 2,    # number of workers for data loading
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=image_processor,                     # The image processor acts as a tokenizer for ViT
    data_collator=collate_fn,                      # Custom collate function for batching
    compute_metrics=compute_metrics,               # Custom metrics function
)

print("Starting training...")
train_results = trainer.train()
print("Training complete!")
# --- 7. Make Predictions on a New Image ---
def predict_image_class(image_path, model, image_processor, id_to_label):
    image = Image.open(image_path).convert("RGB")

    # Preprocess the image
    # Note: Use the validation transforms (no augmentation) for inference
    val_transform = Compose([
        Resize((image_processor.size["height"], image_processor.size["width"])),
        ToTensor(),
        Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
    ])
    
    pixel_values = val_transform(image)
    
    # Add a batch dimension
    inputs = pixel_values.unsqueeze(0)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = inputs.to(device)

    # Perform inference
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        logits = outputs.logits

    # Get predicted class and confidence
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class_label = id_to_label[predicted_class_idx]

    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
    confidence = probabilities[predicted_class_idx].item() * 100

    print(f"\n--- Prediction for {image_path} ---")
    print(f"Predicted class: {predicted_class_label}")
    print(f"Confidence: {confidence:.2f}%")

    # Display image with prediction
    plt.imshow(image)
    plt.title(f"Predicted: {predicted_class_label} ({confidence:.2f}%)")
    plt.axis('off')
    plt.show()

# Example: Pick a random image from the validation set to test
sample_image_path = val_dataset[0]['image'].filename # Get the path of the first validation image
predict_image_class(sample_image_path, model, image_processor, id_to_label)

# You can also try with a custom path
# predict_image_class("path/to/your/new_crop_image.jpg", model, image_processor, id_to_label)
#'''
