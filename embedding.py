import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("combined_emotion[1].csv")
df.drop_duplicates(inplace=True)
df = df.head(10000)

print(df.head(10))
print("value_count: ", df["emotion"].value_counts())
print("----" * 10)

# Map labels to integers
classes = df["emotion"].unique()
label2id = {label: idx for idx, label in enumerate(classes)}
df["label"] = df["emotion"].map(label2id)

# -------------------------------
# Tokenizer
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


# -------------------------------
# Dataset class
# -------------------------------
class EmotionDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len=50):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        tokenized = self.tokenizer(sentence,
                                   padding='max_length',
                                   truncation=True,
                                   max_length=self.max_len,
                                   return_tensors='pt')
        input_ids = tokenized['input_ids'].squeeze(0)
        return input_ids, torch.tensor(label)


# -------------------------------
# DataLoader
# -------------------------------
dataset = EmotionDataset(df['sentence'].tolist(), df['label'].tolist(), tokenizer)
batch_size = 16
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# -------------------------------
# Model
# -------------------------------
class EmotionClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        _, (h_n, _) = self.lstm(x)
        logits = self.linear(h_n[-1])
        return logits


model = EmotionClassifier(vocab_size=tokenizer.vocab_size, embedding_dim=100, hidden_size=64, num_classes=len(classes))

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# -------------------------------
# Training loop with batches
# -------------------------------
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for batch_input_ids, batch_labels in loader:
        optimizer.zero_grad()
        logits = model(batch_input_ids)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch_labels).float().mean()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss / len(loader):.4f} - Acc: {epoch_acc / len(loader):.4f}")
