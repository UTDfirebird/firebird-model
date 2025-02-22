import os
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer

# download necessary NLTK resources
nltk.download('punkt')

def clean(text):
    text = text.lower() # set to lowercase
    hashtags = re.findall(r"#\w+", text)  # extract hashtags
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+", "", text)  # remove mentions (@username)
    text = re.sub(r"[^a-zA-Z0-9\s#]", "", text)  # keep words and hashtags
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text, hashtags

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")    # declare tokenizer

def tokenizeBERT(text): # tokenization function
    return tokenizer.tokenize(text)  # uses BERT’s tokenization method

# function to process a folder of TSVs
def processTSVs(inputFolder, outputFolder):
    # check if output directory exists
    os.makedirs(outputFolder, exist_ok=True)

    # list TSV files in the input folder
    tsvFiles = [f for f in os.listdir(inputFolder) if f.endswith(".tsv")]

    for file in tsvFiles:
        print(f"Processing {file}...")

        # load file
        df = pd.read_csv(os.path.join(inputFolder, file), sep="\t")

        # clean tweets
        df[["cleaned_tweet", "hashtags"]] = df["tweet_text"].apply(lambda x: pd.Series(clean(x)))

        # tokenize using BERT
        df["tokenized_tweet"] = df["cleaned_tweet"].apply(tokenizeBERT)

        # save processed data
        outputPath = os.path.join(outputFolder, file.replace("train.tsv", "processed.csv"))
        df.to_csv(outputPath, index=False)
        print(f"Saved processed file: {outputPath}\n")

inputFolder = r"C:\Users\Creed\OneDrive\Desktop\Firebird\RawTrainingTSV"
outputFolder = r"C:\Users\Creed\OneDrive\Desktop\Firebird\ProcessedCSV"

processTSVs(inputFolder, outputFolder)