from google.colab import drive
import torch
from transformers import DistilBertForSequenceClassification, AutoTokenizer

# mount google drive
drive.mount('/content/drive')
# define paths
model_path = "/content/drive/MyDrive/Model Development/TrainedModels/disaster_classifier.pt"
tokenizer_name = "distilbert-base-uncased"

# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load trained model
model = DistilBertForSequenceClassification.from_pretrained(tokenizer_name, num_labels=3)  # 3 classes
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define class labels (make sure these match your training labels)
class_labels = ["wildfire", "hurricane", "earthquake"]


# function for classifying text input
def classify_tweet(text):
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits.squeeze().cpu().tolist()
    confidence = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze().cpu().tolist()

    print(f"\nRaw Logits: {logits}")
    print(f"Confidence Scores: {confidence}\n")

    prediction = torch.argmax(outputs.logits, dim=1).item()
    return class_labels[prediction]


# Interactive loop
print("\Disaster Classification Model")
print("Type a tweet below and see its classification (Type 'exit' to quit)\n")

while True:
    user_input = input("Enter tweet: ").strip()
    if user_input.lower() == "exit":
        print("Exiting...")
        break

    category = classify_tweet(user_input)
    print(f"Predicted Category: {category}\n")
