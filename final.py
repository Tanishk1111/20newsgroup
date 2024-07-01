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

file_path = Path(r'C:\Users\ASUS\Downloads\history.pkl')
with file_path.open('rb') as file:
    data = pickle.load(file)

print(data)
