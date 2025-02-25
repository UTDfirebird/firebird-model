import os
import torch
import random
from torch.utils.data import TensorDataset
from google.colab import drive
from collections import Counter

# mount google drive
drive.mount('/content/drive')

# define directories
baseDir = "/content/drive/MyDrive/Model Development/"
tensorDir = os.path.join(baseDir, "SelectiveTensors")  # input directory with _train.pt files
outputDir = os.path.join(baseDir, "FinalTensors")
os.makedirs(outputDir, exist_ok=True)

# define disaster categories and label mapping (assumed to be set during CSV processing)
disaster_categories = ["wildfire", "hurricane", "earthquake"]

# dictionary to store data per category (as lists of tuples: (input_ids, attention_mask, label))
disaster_data = {category: [] for category in disaster_categories}

# process each file in the tensor directory
for fileName in os.listdir(tensorDir):
    # only process files that end exactly with "_train.pt" (skip those ending with "_train_train.pt")
    if not (fileName.endswith("_train.pt") and not fileName.endswith("_train_train.pt")):
        continue

    filePath = os.path.join(tensorDir, fileName)
    print(f"Loading: {filePath}")
    dataset = torch.load(filePath)  # Load the TensorDataset

    # determine disaster category based on filename (using lowercase)
    for category in disaster_categories:
        if category in fileName.lower():
            # convert TensorDataset into a list of tuples and add to that category
            disaster_data[category].extend(list(zip(dataset.tensors[0], dataset.tensors[1], dataset.tensors[2])))
            break

# define the desired sample size per disaster type
sample_size = 2500  # adjust as needed

balanced_data = []

# for each disaster category, sample exactly sample_size examples if available; otherwise, use all
for category in disaster_categories:
    data_list = disaster_data[category]
    total_samples = len(data_list)
    print(f"{category.upper()} total samples: {total_samples}")
    if total_samples >= sample_size:
        selected = random.sample(data_list, sample_size)
        print(f"Selected {sample_size} samples from {category.upper()}")
    else:
        selected = data_list
        print(f"Using all {total_samples} samples from {category.upper()}")
    balanced_data.extend(selected)

# shuffle the combined data to ensure random order across disaster types
random.shuffle(balanced_data)

# convert the list of tuples back into a TensorDataset
input_ids = torch.stack([item[0] for item in balanced_data])
attention_masks = torch.stack([item[1] for item in balanced_data])
labels = torch.stack([item[2] for item in balanced_data])
combined_dataset = TensorDataset(input_ids, attention_masks, labels)

# save the combined balanced dataset
save_path = os.path.join(outputDir, "balanced_all_disasters.pt")
torch.save(combined_dataset, save_path)
print(f"\nBalanced dataset saved at: {save_path}")
print(f"Total samples in balanced dataset: {len(combined_dataset)}")

# load the balanced dataset to validate the label distribution
balanced_dataset = torch.load(save_path)
all_labels = balanced_dataset.tensors[2].tolist()

# count label frequencies
label_counts = Counter(all_labels)

# map numeric labels to disaster names; adjust if needed.
label_mapping = {0: "wildfire", 1: "hurricane", 2: "earthquake"}

print("\nFinal Balanced Dataset Label Distribution:")
for label, count in label_counts.items():
    label_name = label_mapping.get(label, "unknown")
    print(f"{label_name}: {count}")
