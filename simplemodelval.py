import os
import joblib
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from gensim.models import Word2Vec
from google.colab import drive
from transformers import AutoTokenizer

# mount rive
drive.mount('/content/drive')

# define paths
base_dir = "/content/drive/MyDrive/Model Development/Data Revamp"
simpler_model_dir = os.path.join(base_dir, "SimplerModels")
word2vec_model_path = os.path.join(simpler_model_dir, "word2vec.model")
val_path = os.path.join(base_dir, "Usage CSVs", "test.csv")

# load validation data
val_df = pd.read_csv(val_path)
print(f"Validation dataset loaded with {len(val_df)} samples")

# load Word2Vec model
word2vec_model = Word2Vec.load(word2vec_model_path)

# function to convert text to Word2Vec vectors
def tweet_to_vec(tweet):
    words = tweet.split()
    word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(100)

# prepare validation text and labels
validation_texts = val_df["text"].tolist()
validation_labels = val_df["label"].tolist()

# convert validation texts to Word2Vec embeddings
X_val = np.array([tweet_to_vec(tweet) for tweet in validation_texts])
y_val = np.array(validation_labels)

# convert y_val (validation labels) to numeric if they are strings
label_map = {"wildfire": 0, "hurricane": 1, "earthquake": 2, "non-disaster": 3}
y_val_numeric = [label_map[label] for label in y_val]

# list of simpler models to evaluate
model_names = ["xgboost", "logistic_regression", "svm"]

# evaluate each model
for name in model_names:
    print(f"\nEvaluating {name.replace('_', ' ').title()}...")

    # load the trained model
    model = joblib.load(os.path.join(simpler_model_dir, f"{name}.joblib"))

    # predictions
    y_pred = model.predict(X_val)

    # convert predicted labels to numeric
    y_pred_numeric = y_pred

    # compute metrics
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_val_numeric, y_pred_numeric, average="weighted")
    auc_score = roc_auc_score(pd.get_dummies(y_val_numeric), pd.get_dummies(y_pred_numeric), multi_class="ovr")

    # print evaluation results
    print(f"{name.replace('_', ' ').title()} Validation Results:")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}, AUC: {auc_score:.4f}")

    # confusion Matrix
    conf_matrix = confusion_matrix(y_val_numeric, y_pred_numeric)
    print(f"\nConfusion Matrix ({name.replace('_', ' ').title()}):")
    print(pd.DataFrame(conf_matrix, index=["wildfire", "hurricane", "earthquake", "non-disaster"], columns=["wildfire", "hurricane", "earthquake", "non-disaster"]))
