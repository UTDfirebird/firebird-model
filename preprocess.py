import os
import torch
import pandas as pd
from transformers import DistilBertTokenizer
from torch.utils.data import TensorDataset
from google.colab import drive
from google.cloud import language_v1
from google.oauth2 import service_account
from tqdm import tqdm  # progress bar

# mount drive
drive.mount('/content/drive')

# paths
base_dir = "/content/drive/MyDrive/Model Development/Data Revamp"
csv_dir = os.path.join(base_dir, "Usage CSVs")
tensor_dir = os.path.join(base_dir, "Usage Tensors/LocationRemoved")
os.makedirs(tensor_dir, exist_ok=True)

# csv paths
train_csv_path = os.path.join(csv_dir, "train.csv")
val_csv_path = os.path.join(csv_dir, "val.csv")

# gcp setup
creds_path = "/content/drive/MyDrive/Model Development/nlpCredential.json"
credentials = service_account.Credentials.from_service_account_file(creds_path)
nlp_client = language_v1.LanguageServiceClient(credentials=credentials)

# tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# labels
label_map = {"wildfire": 0, "hurricane": 1, "earthquake": 2, "non-disaster": 3}

def remove_location_entities(text):
    """Remove location-type entities from text using GCP NLP"""
    try:
        document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
        response = nlp_client.analyze_entities(document=document, encoding_type=language_v1.EncodingType.UTF8)

        # collect location spans
        locations = []
        for entity in response.entities:
            if entity.type_ == language_v1.Entity.Type.LOCATION:
                for mention in entity.mentions:
                    start = mention.text.begin_offset
                    end = start + len(mention.text.content)
                    locations.append((start, end))

        # sort and remove
        locations.sort(reverse=True)
        for start, end in locations:
            text = text[:start] + text[end:]

    except Exception as e:
        print(f"NLP error: {e}")
    return text

def load_and_tokenize(csv_path, split_name):
    df = pd.read_csv(csv_path)
    df["text"] = df["text"].astype(str).str.replace('"', '', regex=False)

    clean_texts = []
    edited_count = 0

    print(f"Cleaning entities for {split_name}...")
    for text in tqdm(df["text"], desc=f"Cleaning {split_name}"):
        cleaned = remove_location_entities(text)
        if cleaned != text:
            edited_count += 1
        clean_texts.append(cleaned)

    df["clean_text"] = clean_texts
    labels = df["label"].map(label_map).tolist()
    encodings = tokenizer(df["clean_text"].tolist(), padding="max_length", truncation=True, max_length=128, return_tensors="pt")

    print(f"{split_name}: {edited_count} out of {len(df)} tweets had location data removed ({edited_count/len(df)*100:.2f}%)")
    return TensorDataset(encodings["input_ids"], encodings["attention_mask"], torch.tensor(labels))

# tokenize and save
train_dataset = load_and_tokenize(train_csv_path, "Train Set")
val_dataset = load_and_tokenize(val_csv_path, "Validation Set")

torch.save(train_dataset, os.path.join(tensor_dir, "train_tensor_nolocation.pt"))
torch.save(val_dataset, os.path.join(tensor_dir, "val_tensor_nolocation.pt"))

print("\nLocation-removed tensors saved:")
print(f"Train Tensor: {len(train_dataset)} samples")
print(f"Val Tensor: {len(val_dataset)} samples")
