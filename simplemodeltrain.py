!pip install --quiet gensim
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from gensim.models import Word2Vec
from google.colab import drive

# mount drive
drive.mount('/content/drive')

# define paths
base_dir = "/content/drive/MyDrive/Model Development/Data Revamp"
csv_dir = os.path.join(base_dir, "Usage CSVs")
output_model_dir = os.path.join(base_dir, "SimplerModels")
os.makedirs(output_model_dir, exist_ok=True)

train_path = os.path.join(csv_dir, "train.csv")
val_path = os.path.join(csv_dir, "val.csv")

# load train and validation datasets
train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
print(f"Train loaded with {len(train_df)} samples")
print(f"Validation loaded with {len(val_df)} samples")

# convert labels to numbers
label_map = {"wildfire": 0, "hurricane": 1, "earthquake": 2, "non-disaster": 3}
train_df["label"] = train_df["label"].map(label_map)
val_df["label"] = val_df["label"].map(label_map)

# display category counts
print("Train category distribution:")
print(train_df["label"].value_counts())
print("\nValidation category distribution:")
print(val_df["label"].value_counts())

# train Word2Vec model on the combined train+val data
combined_text = pd.concat([train_df["text"], val_df["text"]], ignore_index=True)
sentences = [tweet.split() for tweet in combined_text]
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# save the Word2Vec model
word2vec_model_path = os.path.join(output_model_dir, "word2vec.model")
word2vec_model.save(word2vec_model_path)
print(f"Word2Vec model saved at: {word2vec_model_path}")

# function to convert text to average word embeddings
def tweet_to_vec(tweet):
    words = tweet.split()
    word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    if not word_vectors:
        return np.zeros(100)
    return np.mean(word_vectors, axis=0)

# convert text to vectors
X_train = np.array([tweet_to_vec(tweet) for tweet in train_df["text"]])
y_train = train_df["label"].values

X_val = np.array([tweet_to_vec(tweet) for tweet in val_df["text"]])
y_val = val_df["label"].values

# define models
models = {
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"),
    "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
    "SVM": SVC(kernel="linear", probability=True, class_weight="balanced")
}

# train and evaluate models
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, target_names=label_map.keys())

    print(f"{name} Accuracy: {acc:.4f}")
    print(report)

    results[name] = {
        "accuracy": acc,
        "classification_report": report
    }

    # save model
    model_path = os.path.join(output_model_dir, f"{name.replace(' ', '_').lower()}.joblib")
    joblib.dump(model, model_path)
    print(f"{name} model saved at {model_path}")

print("\nTraining and evaluation completed.")
