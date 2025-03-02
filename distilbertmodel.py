from google.colab import drive
import os
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW
from transformers import get_scheduler
from collections import Counter

# mount drive
drive.mount('/content/drive')

# define dataset paths
baseDir = "/content/drive/MyDrive/Model Development/FinalTensors/"
augmentedDatasetPath = os.path.join(baseDir, "augmented_balanced_all_disasters.pt")

# load the augmented dataset
trainData = torch.load(augmentedDatasetPath)

# define batch size (adjust based on GPU memory)
batch_size = 16

# create DataLoader for training, enable shuffling
trainLoader = DataLoader(trainData, batch_size=batch_size, shuffle=True)

# print dataset size and number of batches
print(f"Total Training Samples: {len(trainData)}")
print(f"Total Training Batches: {len(trainLoader)}")

# count and print the number of samples per label
labelList = trainData.tensors[2].tolist()
counts = Counter(labelList)
print("\nTraining Data Label Counts:")
labelMapping = {0: "wildfire", 1: "hurricane", 2: "earthquake", 3: "non-disaster"}
for label, count in counts.items():
    print(f"{labelMapping.get(label, 'unknown')}: {count}")

# update number of classes to 4
num_labels = 4

# load DistilBERT with a classification head
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

# move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("\nModel loaded and moved to:", device)
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU found, training will be on CPU.")

# define optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
num_training_steps = len(trainLoader) * 3  # Assuming 3 epochs
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# define loss function (CrossEntropyLoss for multi-class classification)
loss_fn = torch.nn.CrossEntropyLoss()

# training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in trainLoader:
        # unpack batch and move to GPU
        input_ids, attention_mask, labels = [t.to(device) for t in batch]

        # forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(trainLoader)
    print(f"Epoch {epoch+1}: Training Loss = {avg_loss:.4f}")

# ensure the output directory exists
output_model_dir = "/content/drive/MyDrive/Model Development/TrainedModels"
os.makedirs(output_model_dir, exist_ok=True)

# move model to CPU before saving
model.cpu()
output_model_path = os.path.join(output_model_dir, "disaster_classifierV2.pt")
torch.save(model.state_dict(), output_model_path)
print(f"Model saved successfully to {output_model_path}")

# move back to GPU if needed
model.to(device)
