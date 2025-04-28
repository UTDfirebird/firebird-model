from google.colab import drive
import os
import torch
import pandas as pd
import numpy as np
from transformers import DistilBertForSequenceClassification, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset  # import TensorDataset here
from gensim.models import Word2Vec

# mount drive
drive.mount('/content/drive')

# define updated paths
base_dir = "/content/drive/MyDrive/Model Development"
distilbert_model_path = os.path.join(base_dir, "TrainedModels", "disaster_classifierV5.pt")
validation_tensor_path = os.path.join(base_dir, "Data Revamp", "Usage Tensors", "test_tensor.pt")

# load tokenizer and DistilBERT model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

distilbert_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4)
distilbert_model.load_state_dict(torch.load(distilbert_model_path, map_location=device))
distilbert_model.to(device)
distilbert_model.eval()

# load new validation tensor
torch.serialization.add_safe_globals([TensorDataset])
validation_data = torch.load(validation_tensor_path, weights_only=False)
val_loader = DataLoader(validation_data, batch_size=16, shuffle=False)

# class labels
class_labels = ["wildfire", "hurricane", "earthquake", "non-disaster"]

# evaluation
all_preds, all_labels, all_confidences = [], [], []
print("\nRunning DistilBERT on validation set...")

with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, labels = [t.to(device) for t in batch]
        outputs = distilbert_model(input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)

        predictions = torch.argmax(logits, dim=1)
        confidences = probs.max(dim=1).values

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_confidences.extend(confidences.cpu().numpy())

# DistilBERT metrics
precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted")
auc_score = roc_auc_score(pd.get_dummies(all_labels), pd.get_dummies(all_preds), multi_class="ovr")

print("\nDistilBERT Validation Results:")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}, AUC: {auc_score:.4f}")
print("\nConfusion Matrix (DistilBERT):")
print(pd.DataFrame(confusion_matrix(all_labels, all_preds), index=class_labels, columns=class_labels))
