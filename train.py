import keras
import pandas as pd
import numpy as np
import re
import os
import joblib
from pathlib import Path
from scipy.stats import wilcoxon

import torch
# Text and feature engineering
from sklearn.feature_extraction.text import TfidfVectorizer

# Evaluation and tuning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc)

# Classifier
from classifier import ConvolutionalAttention
from skorch import NeuralNetClassifier

# Text cleaning & stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
torch.serialization.add_safe_globals([ConvolutionalAttention])

# Variable that determines if we're training on all datasets to give the tool as much data as possible,
# or if we just want to test metrics for a specific project.
use_all_projects = False

########## 2. Define text preprocessing methods ##########

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

def collate_fn(batch):
    X, y = zip(*batch)
    X = torch.tensor(np.stack(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    return X, y

########## 3. Download & read data ##########
import os
import subprocess

if __name__ == "__main__":
    # Choose the project (options: 'pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe')
    pd_all = None
    project = 'tensorflow'
    positive_weight = 0 # weight for positive values, since there's a class imbalance!

    if use_all_projects:
        print("Using all projects for training...")
        concat_df = None
        paths = Path('baseline_model/datasets').rglob('*.csv')
        for path in paths:
            pathstr = str(path)
            df = pd.read_csv(path)
            df = df.sample(frac=1, random_state=999)
            if concat_df is None:
                concat_df = df
            else:
                concat_df = pd.concat([concat_df, df])
        pd_all = concat_df
        positive_weight = 6.098
    else:
        print(f"Using {project} project for training...")
        path = f'baseline_model/datasets/{project}.csv'
        df = pd.read_csv(path)
        pd_all = df.sample(frac=1, random_state=999)  # Shuffle

        # This sets weight as 100 / percentage value as given in lab1 PDF.
        if project == 'tensorflow':
            positive_weight = 5.348
        elif project == 'keras':
            positive_weight = 4.950
        elif project == 'pytorch':
            positive_weight = 7.937
        elif project == 'incubator-mxnet':
            positive_weight = 7.937
        elif project == 'caffe':
            positive_weight = 8.696

    # Merge Title and Body into a single column; if Body is NaN, use Title only
    pd_all['Title+Body'] = pd_all.apply(
        lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
        axis=1
    )

    # Keep only necessary columns: id, Number, sentiment, text (merged Title+Body)
    pd_tplusb = pd_all.rename(columns={
        "Unnamed: 0": "id",
        "class": "sentiment",
        "Title+Body": "text"
    })
    pd_tplusb.to_csv('Title+Body.csv', index=False, columns=["id", "Number", "sentiment", "text"])

    ########## 4. Configure parameters & Start training ##########

    # ========== Key Configurations ==========

    # 1) Data file to read
    datafile = 'Title+Body.csv'

    # 2) Number of repeated experiments
    REPEAT = 30

    # 3) Output CSV file name
    out_csv_name = f'../{project}_NB.csv'

    # ========== Read and clean data ==========
    data = pd.read_csv(datafile).fillna('')
    text_col = 'text'

    # Keep a copy for referencing original data if needed
    original_data = data.copy()

    # Text cleaning
    data[text_col] = data[text_col].apply(remove_html)
    data[text_col] = data[text_col].apply(remove_emoji)
    data[text_col] = data[text_col].apply(remove_stopwords)
    data[text_col] = data[text_col].apply(clean_str)

    # ========== Hyperparameter grid ==========
    # We use logspace for var_smoothing: [1e-12, 1e-11, ..., 1]
    params = {
        'module__num_channels': [32, 64],
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Lists to store metrics across repeated runs
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    auc_values = []

    classifier_save = None

    for repeated_time in range(REPEAT):
        # --- 4.1 Split into train/test ---
        indices = np.arange(data.shape[0])
        train_index, test_index = train_test_split(
            indices, test_size=0.2, random_state=repeated_time
        )

        # --- 4.2 Tokenisation ---
        text_vectoriser = keras.layers.TextVectorization(max_tokens=10000, output_sequence_length=50)
        text_vectoriser.adapt(data[text_col])
        vocab_size = len(text_vectoriser.get_vocabulary())
        X_all = text_vectoriser(data[text_col]).numpy()
        X_train = X_all[train_index]
        X_test  = X_all[test_index]
        y_train = data['sentiment'].iloc[train_index]
        y_test  = data['sentiment'].iloc[test_index]

        # --- 4.3 Convolutional Attention model & GridSearch ---
        clf = NeuralNetClassifier(
            module=ConvolutionalAttention,
            module__vocab_size=vocab_size,
            module__num_channels=64,
            module__num_classes=1,
            module__fc_dim=64,
            max_epochs=20,
            lr=0.001,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            iterator_train__collate_fn=collate_fn,
            iterator_valid__collate_fn=collate_fn,
            criterion=torch.nn.BCEWithLogitsLoss,
            criterion__pos_weight=torch.Tensor([positive_weight]), # this is 100 / percentage from lab1 PDF
            optimizer=torch.optim.Adam,
        )

        grid = GridSearchCV(
            clf,
            params,
            cv=5,              # 5-fold CV (can be changed)
            scoring='roc_auc'  # Using roc_auc as the metric for selection
        )
        grid.fit(X_train, y_train)

        # Retrieve the best model
        best_clf = grid.best_estimator_
        best_clf.fit(X_train, y_train)

        # --- 4.4 Make predictions & evaluate ---
        y_pred = best_clf.predict(X_test)

        # Accuracy
        acc = accuracy_score(y_test, y_pred)

        # Saving the classifier with the highest accuracy
        if len(accuracies) == 0: classifier_save = best_clf
        if all(i <= acc for i in accuracies): classifier_save = best_clf

        accuracies.append(acc)

        # Precision (macro)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        precisions.append(prec)

        # Recall (macro)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        recalls.append(rec)

        # F1 Score (macro)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_scores.append(f1)

        # AUC
        # If labels are 0/1 only, this works directly.
        # If labels are something else, adjust pos_label accordingly.
        fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=1)
        auc_val = auc(fpr, tpr)
        auc_values.append(auc_val)

    # --- 4.5 Aggregate results ---
    final_accuracy  = np.mean(accuracies)
    final_precision = np.mean(precisions)
    final_recall    = np.mean(recalls)
    final_f1        = np.mean(f1_scores)
    final_auc       = np.mean(auc_values)

    csv_file = f"baseline_model/nb_results_{project}.csv"
    nb_df = pd.read_csv(csv_file)
    nb_accuracies = nb_df["accuracies"].values
    nb_fmeasures = nb_df["fmeasures"].values
    nb_aucs = nb_df["aucs"].values

    # Calculate wilcoxon signed rank values
    accuracy_stat, accuracy_p = wilcoxon(accuracies, nb_accuracies)
    fmeasure_stat, fmeasure_p = wilcoxon(f1_scores, nb_fmeasures)
    auc_stat, auc_p = wilcoxon(auc_values, nb_aucs)

    scores = pd.DataFrame({
        "index": np.arange(1, len(accuracies) + 1),
        "accuracies": accuracies,
        "fmeasures": f1_scores,
        "aucs": auc_values
    })

    csv_file = f"cnn_results_{project}.csv"
    scores.to_csv(csv_file, index=False)
    print("Saved results.")

    print("=== Attention-CNN + Tokenisation Results ===")
    print(f"Number of repeats:     {REPEAT}")
    print(f"Average Accuracy:      {final_accuracy:.4f}")
    print(f"Average Precision:     {final_precision:.4f}")
    print(f"Average Recall:        {final_recall:.4f}")
    print(f"Average F1 score:      {final_f1:.4f}")
    print(f"Average AUC:           {final_auc:.4f}")

    print("=== Wilcoxon Signed-Rank Test Results ===")
    print(f"Accuracy: statistic = {accuracy_stat:.4f}, p-value = {accuracy_p:.4f}")
    print(f"F-Measure: statistic = {fmeasure_stat:.4f}, p-value = {fmeasure_p:.4f}")
    print(f"ROC AUC: statistic = {auc_stat:.4f}, p-value = {auc_p:.4f}")

    # Saving Trained Model #
    if use_all_projects: # only want to do it when building model for GUI
        print("Saving model...")
        filename = "model.joblib.pkl"
        _ = joblib.dump(classifier_save, filename)
        print("Saving vectoriser...")
        vectoriser_filename = "vectoriser.joblib.pkl"
        _ = joblib.dump(text_vectoriser, vectoriser_filename)