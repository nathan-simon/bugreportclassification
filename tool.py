import builtins
import collections

import joblib
import re
import tkinter as tk
from tkinter import messagebox
from train import *
import torch.nn as nn
from classifier import ResNetBlock, Nin

from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Stuff so that PyTorch doesn't complain about unpickling
torch.serialization.add_safe_globals([nn.Linear])
torch.serialization.add_safe_globals([nn.Conv1d])
torch.serialization.add_safe_globals([nn.ModuleList])
torch.serialization.add_safe_globals([ResNetBlock])
torch.serialization.add_safe_globals([nn.Embedding])
torch.serialization.add_safe_globals([nn.Dropout])
torch.serialization.add_safe_globals([torch.optim.Adam])
torch.serialization.add_safe_globals([nn.MultiheadAttention])
torch.serialization.add_safe_globals([nn.modules.linear.NonDynamicallyQuantizableLinear])
torch.serialization.add_safe_globals([Nin])
torch.serialization.add_safe_globals([nn.BCEWithLogitsLoss])
torch.serialization.add_safe_globals([torch.optim.SGD])
torch.serialization.add_safe_globals([collections.defaultdict])
torch.serialization.add_safe_globals([builtins.dict])

clf = None
vectoriser = None

def load_model(file):
    global clf
    clf = joblib.load(file)

def load_vectoriser(file):
    global vectoriser
    vectoriser = joblib.load(file)

def remove_html(text):
    """Remove HTML tags using a regex."""
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_emoji(text):
    """Remove emojis using a regex pattern."""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"  # enclosed characters
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Stopwords
NLTK_stop_words_list = stopwords.words('english')
custom_stop_words_list = ['...']  # You can customize this list as needed
final_stop_words_list = NLTK_stop_words_list + custom_stop_words_list

def remove_stopwords(text):
    """Remove stopwords from the text."""
    return " ".join([word for word in str(text).split() if word not in final_stop_words_list])

def clean_str(string):
    """
    Clean text by removing non-alphanumeric characters,
    and convert it to lowercase.
    """
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

def predict(title, description):
    text = (title + ". " + description).lower()
    text = remove_html(text)
    text = remove_emoji(text)
    text = remove_stopwords(text)
    text = clean_str(text)
    inputs = vectoriser([text]).numpy()
    pred = clf.predict(inputs)
    return pred[0] == 1

def on_submit():
    title = title_entry.get()
    description = description_text.get("1.0", tk.END).strip()

    if not title or not description:
        messagebox.showwarning("Input Error", "Please provide both a title and a description.")
        return

    is_performance = predict(title, description)
    result = "Bug report is performance related." if is_performance else "Bug report is NOT performance related."

    result_label.config(text=result)

# Load the pre-trained CNN-attention model.
filename = "model.joblib.pkl"
load_model(filename)
vectoriser_filename = "vectoriser.joblib.pkl"
load_vectoriser(vectoriser_filename)

root = tk.Tk()
root.title("Bug Report Analyser")

title_label = tk.Label(root, text="Bug Report Title:")
title_label.pack(pady=(10, 0))

title_entry = tk.Entry(root, width=50)
title_entry.pack(padx=10, pady=(0, 10))

description_label = tk.Label(root, text="Bug Report Description:")
description_label.pack()

description_text = tk.Text(root, width=50, height=10, font="Arial")
description_text.pack(padx=10, pady=(0, 10))

submit_button = tk.Button(root, text="Analyse", command=on_submit)
submit_button.pack(pady=(0, 10))

result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack(pady=(0, 10))

root.mainloop()