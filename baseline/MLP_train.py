import pickle
import os
import pandas as pd
import numpy as np
import torch

btc_data = pd.read_feather("/content/drive/MyDrive/2025.01 비트코인 선물 봇 기본 실험 데이터/BTC_USDT-1d.feather") # chart ranges from 20190101-20250123
filtered_btc_data = pd.read_csv("/content/drive/MyDrive/2025.01 비트코인 선물 봇 기본 실험 데이터/baseline_experiments/filtered_btc_data_with_labels_for_baseline_train.csv") # data with labels 

with open("/content/drive/MyDrive/2025.01 비트코인 선물 봇 기본 실험 데이터/baseline_experiments/baseline_grouped_news.pkl", "rb") as f:
  loaded_grouped_news = pickle.load(f)

embedding = torch.load("/content/drive/MyDrive/2025.01 비트코인 선물 봇 기본 실험 데이터/baseline_experiments/grouped_news_embeddings.pt", weights_only=True)

print(embedding.shape)

min_date = min(loaded_grouped_news.keys())
max_date = max(loaded_grouped_news.keys())
print(f"Date range in grouped_news: {min_date} to {max_date}") # full data ranges from 20190324-20250124


labels = filtered_btc_data["label"].values
embedding = embedding[:-2, :] # exclude 2025-01-23, 2025-01-24 

# train/val/test split, chronological 
train_size = int(0.8 * len(embedding))
val_size = int(0.1 * len(embedding))
test_size = len(embedding) - train_size - val_size

train_embeddings = embedding[:train_size]
train_labels = labels[:train_size]

val_embeddings = embedding[train_size:train_size+val_size]
val_labels = labels[train_size:train_size+val_size]

test_embeddings = embedding[train_size+val_size:]
test_labels = labels[train_size+val_size:]

print(train_embeddings.shape, train_labels.shape, val_embeddings.shape, val_labels.shape, test_embeddings.shape, test_labels.shape) 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(device)

input_dim = train_embeddings.shape[1]
hidden_dim = 512
output_dim = 1 # binary classification (0:long or 1:short)
learning_rate = 1e-4
batch_size = 32 
num_epochs = 50 
checkpoint_path = "best_clf.pt"  # Path to save the best model

class Mish(nn.Module):
  def forward(self, x):
    return x * torch.tanh(nn.functional.softplus(x))

class CLF(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim):
    super(CLF, self).__init__()
    self.fc1 = nn.Linear(input_dim, hidden_dim)
    self.mish1 = Mish()
    self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
    self.mish2 = Mish()
    self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
    self.sigmoid = nn.Sigmoid()
  def forward(self, x):
    x = self.fc1(x)
    x = self.mish1(x)
    x = self.fc2(x)
    x = self.mish2(x)
    x = self.fc3(x)
    x = self.sigmoid(x)
    return x

model = CLF(input_dim, hidden_dim, output_dim).to(device)
criterion = nn.BCELoss() # binary cross entropy loss 
optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

def create_dataloader(embedding_data, labels, batch_size):
  dataset = TensorDataset(torch.tensor(embedding_data, dtype=torch.float32),
                            torch.tensor(labels, dtype=torch.float32).unsqueeze(1))
  return DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_loader = create_dataloader(train_embeddings, train_labels, batch_size)
val_loader = create_dataloader(val_embeddings, val_labels, batch_size)
test_loader = create_dataloader(test_embeddings, test_labels, batch_size)

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs): 
  best_val_loss = 100000000 
  best_epoch = 0 
  for epoch in range(num_epochs):
    model.train() 
    train_loss = 0.0 
    for embeddings, labels in train_loader: 
      embeddings, labels = embeddings.to(device), labels.to(device) 
      # forward pass 
      outputs = model(embeddings) 
      loss = criterion(outputs, labels)  
      # backward pass and optimization 
      optimizer.zero_grad() 
      loss.backward() 
      optimizer.step() 
      train_loss += loss.item() 
    
    train_loss /= len(train_loader)

    val_loss = 0.0 
    model.eval() 
    with torch.no_grad():
      for embeddings, labels in val_loader:
        embeddings, labels = embeddings.to(device), labels.to(device) 
        outputs = model(embeddings) 
        loss = criterion(outputs, labels) 
        val_loss += loss.item()
    
    val_loss /= len(val_loader) 

    if val_loss < best_val_loss:
      best_val_loss = val_loss 
      torch.save(model.state_dict(), checkpoint_path) 
      best_epoch = epoch + 1 

    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")  

  print(f"Best Epoch: {best_epoch}")

train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs) 

# Load the Best Model for Testing
model.load_state_dict(torch.load(checkpoint_path))
model.eval()
print(f"Loaded best model from checkpoint: {checkpoint_path}")

def test_model(model, test_loader):
  all_predictions = [] 
  all_labels = [] 
  with torch.no_grad():
    for embeddings, labels in test_loader:
      embeddings, labels = embeddings.to(device), labels.to(device) 
      outputs = model(embeddings) 
      predictions = (outputs > 0.5).float() 
      all_predictions.extend(predictions.cpu().numpy())  
      all_labels.extend(labels.cpu().numpy()) 
  
  accuracy = np.mean(np.array(all_predictions) == np.array(all_labels)) 
  print(f"Test Accuracy: {accuracy * 100:.2f}%")

test_model(model, test_loader)


""" 
output: 

Loaded best model from checkpoint: best_clf.pt
Test Accuracy: 53.59%
"""
