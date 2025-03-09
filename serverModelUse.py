import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AutoTokenizer
import os
os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache/huggingface/"

# define paths
path = "model/disaster_classifierV2.pt"  # V2 Model
tokenizer_name = "distilbert-base-uncased"

# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load trained model (updated for 4 classes)
model = DistilBertForSequenceClassification.from_pretrained(tokenizer_name, num_labels=4)  # Now handling 4 categories
model.load_state_dict(torch.load(path, map_location=device))
model.to(device)
model.eval()

# updated class labels (now includes non-disaster)
class_labels = ["wildfire", "hurricane", "earthquake", "non-disaster"]

# tokenize text for processing
def preprocess_text(text):
    return tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)

# classify tweet and return confidence scores
def classify_tweet(text): 
    inputs = preprocess_text(text)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0].cpu().numpy()

    # convert logits to probabilities using softmax
    probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

    # get predicted category
    predicted_class = torch.argmax(torch.tensor(probabilities)).item()

    return predicted_class, probabilities.tolist()

# process data (parse JSON, extract tweets, classify them)
def process_server_data(tweetDict):
    results = {}
    for tweetID, text in tweetDict.items():
        if text:  # ensure the text is not empty
            category, confidence_scores = classify_tweet(text)
            results[tweetID] = confidence_scores
    return results

# process parsed data
#classification_results = process_server_data(parsed_data)  

# print results
#for tweetID, confidence_scores in classification_results.items():
#    print(f"Tweet ID: {tweetID}")
#    print(f"Confidence Scores: {confidence_scores}")
#    print("-" * 50)
