import os
import pandas as pd
import re
import ast
import torch
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer
from google.colab import drive

# miunt google drive
drive.mount('/content/drive')

# define directories
baseDir = "/content/drive/MyDrive/Model Development/"
inputDir = os.path.join(baseDir, "RawTrainingTSV")
outputDir = os.path.join(baseDir, "SelectiveTensors")
os.makedirs(outputDir, exist_ok=True)

# initialize the DistilBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# define a cleaning function for tweet text
def clean_text(text):
    text = str(text).lower()                              # lowercase the text
    text = re.sub(r"http\S+|www\S+", "", text)             # demove URLs
    text = re.sub(r"@\w+", "", text)                       # remove mentions
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)             # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()               # remove extra spaces
    return text

# function to tokenize text using the DistilBERT tokenizer
def tokenize_text(text):
    # tokenize text with padding/truncation to max_length=128
    tokens = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    # return a flattened list of token ids
    return tokens["input_ids"].squeeze().tolist()

# define a label mapping: convert disaster string labels to numeric values
label_mapping = {"wildfire": 0, "hurricane": 1, "earthquake": 2}

# list of disaster keywords to search for in the filename
disaster_keywords = list(label_mapping.keys())

# process each TSV file in the input directory
for fileName in os.listdir(inputDir):
    if not fileName.endswith(".tsv"):
        continue

    # determine disaster type from the filename (case-insensitive)
    file_lower = fileName.lower()
    disaster_type = None
    for keyword in disaster_keywords:
        if keyword in file_lower:
            disaster_type = keyword
            break

    if disaster_type is None:
        print(f"Skipping {fileName} as no disaster type was found in its name.")
        continue

    filePath = os.path.join(inputDir, fileName)
    print(f"Processing: {filePath} as {disaster_type}")

    # read the TSV file (assumes tab-separated values)
    df = pd.read_csv(filePath, delimiter="\t")

    # check for expected columns
    if not {"tweet_id", "tweet_text", "class_label"}.issubset(df.columns):
        print(f"Skipping {fileName} - required columns not found.")
        continue

    # clean the tweet text
    df["cleaned_tweet"] = df["tweet_text"].apply(clean_text)

    # tokenize the cleaned tweet text; store token ids as a list of ints
    df["tokenized"] = df["cleaned_tweet"].apply(tokenize_text)

    # overwrite the class_label with the disaster type from the filename (ensuring lowercase)
    df["class_label"] = disaster_type.lower()
    # map string labels to numeric values using the label_mapping
    df["label"] = df["class_label"].map(label_mapping)

    # convert the tokenized column into a tensor (list of ints per tweet)
    input_ids = torch.tensor(df["tokenized"].tolist())
    # create an attention mask: non-zero tokens are marked with 1
    attention_masks = (input_ids != 0).long()
    # convert labels to a tensor
    labels = torch.tensor(df["label"].tolist())

    # create a TensorDataset from input_ids, attention_masks, and labels
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # define the output file path, changing the extension from .tsv to .pt
    output_file = os.path.join(outputDir, fileName.replace(".tsv", ".pt"))
    torch.save(dataset, output_file)
    print(f"Saved tensor dataset to {output_file} (Total samples: {len(dataset)})")

print("Processing complete. All files saved in the SelectiveTensors folder.")
