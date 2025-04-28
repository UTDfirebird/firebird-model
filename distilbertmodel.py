from google.colab import drive
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

# track performance per epoch
train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []

# mount google drive
drive.mount('/content/drive')

# define dataset paths
base_dir = "/content/drive/MyDrive/Model Development/Data Revamp/Usage Tensors"
tensor_dir = os.path.join(base_dir, "LocationRemoved")
train_tensor_path = os.path.join(tensor_dir, "train_tensor_nolocation.pt")
val_tensor_path = os.path.join(tensor_dir, "val_tensor_nolocation.pt")

from torch.utils.data import TensorDataset
import torch.serialization
torch.serialization.add_safe_globals([TensorDataset])

# load tensors
train_data = torch.load(train_tensor_path, weights_only=False)
val_data = torch.load(val_tensor_path, weights_only=False)

# create DataLoaders
batch_size = 16
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# label mapping
label_mapping = {0: "wildfire", 1: "hurricane", 2: "earthquake", 3: "non-disaster"}

# check training label distribution
label_list = train_data.tensors[2].tolist()
counts = Counter(label_list)
print(f"\nTotal Training Samples: {len(train_data)}")
print(f"Training Batches: {len(train_loader)}")
print("\nTraining Label Counts:")
for label, count in counts.items():
    print(f"{label_mapping[label]}: {count}")

# load model
num_labels = 4
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

# move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"\nModel loaded and moved to: {device}")
if device.type == "cuda":
    print("Using GPU:", torch.cuda.get_device_name(0))

# optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
num_training_steps = len(train_loader) * 3  # 3 epochs
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in train_loader:
        input_ids, attention_mask, labels = [t.to(device) for t in batch]
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        logits = outputs.logits

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    train_acc = accuracy_score(all_labels, all_preds)
    train_loss_history.append(avg_loss)
    train_acc_history.append(train_acc)

    print(f"\nEpoch {epoch+1} Results:")
    print(f"Training Loss: {avg_loss:.4f}")
    print(f"Training Accuracy: {train_acc:.4f}")
    print("Training Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_mapping.values(), digits=4))

    # validation
    model.eval()
    val_preds = []
    val_labels = []
    val_total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [t.to(device) for t in batch]
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            logits = outputs.logits
            val_total_loss += outputs.loss.item()

            val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())


    val_loss = val_total_loss / len(val_loader)
    val_acc = accuracy_score(val_labels, val_preds)
    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc)

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print("Validation Classification Report:")
    print(classification_report(val_labels, val_preds, target_names=label_mapping.values(), digits=4))

    # generalization gap warning
    gap = train_acc - val_acc
    print(f"Generalization Gap: {gap:.4f}")
    if gap > 0.05:
        print("Potential overfitting. Training accuracy significantly exceeds validation accuracy.")

# save final model
model.cpu()
output_model_dir = "/content/drive/MyDrive/Model Development/TrainedModels"
os.makedirs(output_model_dir, exist_ok=True)
output_model_path = os.path.join(output_model_dir, "disaster_classifierNoLocation.pt")
torch.save(model.state_dict(), output_model_path)
print(f"\nModel saved successfully to {output_model_path}")

# plot loss curve
plt.figure(figsize=(10,4))
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Validation Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# plot accuracy curve
plt.figure(figsize=(10,4))
plt.plot(train_acc_history, label="Train Accuracy")
plt.plot(val_acc_history, label="Validation Accuracy")
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
