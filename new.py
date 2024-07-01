# -*- coding: utf-8 -*-
"""20newsgroup.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/14PztWDQzJr7ViaJ3WQsZe7rTzJ1xCtjA
"""


import os
import tarfile
import matplotlib.pyplot as plt
import pandas as pd
import torch
from pathlib import Path
from NeuralNMF import train
from sklearn.feature_extraction.text import TfidfVectorizer
from NeuralNMF import Neural_NMF
import pickle



train_data_dir = Path("/home/Tanishk/20news-bydate/20news-bydate-train")
test_data_dir = Path("/home/Tanishk/20news-bydate/20news-bydate-test")

# Function to load data
def load_20newsgroups(data_dir):
    data = []
    target = []

    for newsgroup in os.listdir(data_dir):
        newsgroup_path = os.path.join(data_dir, newsgroup)

        if os.path.isdir(newsgroup_path):
            for filename in os.listdir(newsgroup_path):
                file_path = os.path.join(newsgroup_path, filename)

                if os.path.isfile(file_path):
                    with open(file_path, 'r', encoding='latin1') as file:
                        content = file.read()
                        data.append(content)
                        target.append(newsgroup)

    return pd.DataFrame({'text': data, 'newsgroup': target})

train_df = load_20newsgroups(train_data_dir)
test_df = load_20newsgroups(test_data_dir)

train_df.head()



newsgroup_counts = train_df['newsgroup'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(newsgroup_counts, labels=newsgroup_counts.index, autopct='%1.1f%%', startangle=90)
_ = plt.title('Distribution of Newsgroup Categories')



tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(train_df['text'])

X = torch.tensor(tfidf.toarray(), dtype=torch.float64)
X.shape

X1 = X[0:11300]
m = 100
k1 = 10
k2 = 5

net = Neural_NMF([m, k1, k2])


def create_batches(X1, batch_size):
    num_samples = 11300
    indices = torch.randperm(num_samples)
    return [X1[indices[i:i + batch_size]] for i in range(0, num_samples, batch_size)]

# Create batches
batch_size = 100  # Adjust batch size as needed
batches = create_batches(X1, batch_size)

history = []

# Function to train on batches
def train_on_batches(net, batches, epochs, lr):
    for batch in batches:
        history1 = train(net, batch, epoch=epochs, lr=lr, supervised=False)
        history.append(history1)


# Train the model on batches
train_on_batches(net, batches, epochs=6, lr=500)

with open("history.pkl", "wb") as file:
    pickle.dump(history, file)