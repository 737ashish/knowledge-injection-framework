import matplotlib.pyplot as plt
import re
from nltk.tokenize.treebank import TreebankWordDetokenizer

def preprocess_text(tokenized_sent):
    """
    in some cases TreebankWordDetokenizer still does not detokenize sentence correctly
    for those sentences ie extraction and generation performance seems worse
    """
    sent = TreebankWordDetokenizer().detokenize(tokenized_sent)
    sent = sent.replace(" - ", "-").replace(" – ", "-")
    sent = sent.replace("’", "'")
    sent = sent.replace(" 's", "'s")
    sent = sent.replace(" / ", "/")
    sent = sent.replace("\"", "")
    sent = re.sub(r"\(.*?\)", "", sent) # remove unnecessary text in parentheses
    sent = re.sub(r"\[.*?\]", "", sent) # remove unnecessary text in brackets
    sent = re.sub(r"\s*,\s*", ", ", sent) # remove all spaces before comma
    sent = sent.replace(" .", ".")
    sent = " ".join(sent.split()) # remove multiple spaces
    return sent


def ireplace(text, pattern, replacement):
    """Case-insensitive replace."""
    return re.sub("(?i)" + re.escape(pattern), replacement, text)


def dataset_stats(df):
    """Plot dataset statistics."""
    print(f"Unique relations: {df['r_id'].nunique()}")
    print(f"Num samples: {len(df)}")
    
    plt.title("Num samples per relation")
    df["r"].value_counts().plot(kind="bar", figsize=(15, 3))
    plt.xlabel("relation")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.show()
