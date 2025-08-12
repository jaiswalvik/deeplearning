import torch
import torch.nn as nn
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# 1. Set seed
torch.manual_seed(42)

# 2. Load data
df = pd.read_csv("preprocessed_yelp_data.csv")  # after downloading from your link

# 3. Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(len(train_df), len(test_df))  # First Q: dataset sizes

# 4. Build vocabulary
def build_vocab(texts, max_size=10000, min_freq=2):
    vocab = {"<UNK>": 0, "<PAD>": 1}
    freq = Counter()
    for tokens in texts:
        freq.update(tokens)
    for word, count in freq.items():
        if count >= min_freq and len(vocab) < max_size:
            vocab[word] = len(vocab)
    return vocab

train_texts = train_df['text'].tolist()
train_texts = [ast.literal_eval(t) for t in train_texts]
vocab = build_vocab(train_texts, max_size=10000, min_freq=2)
print(len(vocab))  # vocab size Q

# 5. Numericalize
def numericalize(text, vocab):
    return [vocab.get(token, vocab["<UNK>"]) for token in text]

train_df["text"] = [numericalize(ast.literal_eval(t), vocab) for t in train_df["text"]]
test_df["text"] = [numericalize(ast.literal_eval(t), vocab) for t in test_df["text"]]
print(train_df.iloc[0]["text"])  # numericalized example Q

# 6. YelpDataset class
class YelpDataset(Dataset):
    def __init__(self, dataframe, max_seq_length):
        self.dataframe = dataframe
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.dataframe)

    def pad_or_truncate(self, seq):
        if len(seq) < self.max_seq_length:
            seq = seq + [1] * (self.max_seq_length - len(seq))  # PAD index=1
        else:
            seq = seq[:self.max_seq_length]
        return seq

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx]["text"]
        label = self.dataframe.iloc[idx]["label"]
        text_tensor = torch.tensor(self.pad_or_truncate(text), dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return {"text": text_tensor, "label": label_tensor}

# 7. Create datasets & loaders
max_seq_length = 100
train_dataset = YelpDataset(train_df, max_seq_length)
test_dataset = YelpDataset(test_df, max_seq_length)
print(train_dataset[0])  # tensor example Q

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 8. RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, num_layers, output_size):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        out = self.fc(hidden[-1])
        return out

input_size = len(vocab)
embedding_dim = 100
hidden_size = 256
num_layers = 2
output_size = 1

model = RNNModel(input_size, embedding_dim, hidden_size, num_layers, output_size)
print(sum(p.numel() for p in model.parameters()))  # param count Q

# 9. Loss & optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 10. Train & evaluate
def train_model():
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(batch["text"]).squeeze(1).float()
        labels = batch["label"].float()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model():
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(batch["text"]).squeeze(1).float()
            labels = batch["label"].float()
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.round(torch.sigmoid(outputs))
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(test_loader), (correct / total) * 100

for epoch in range(2):
    train_loss = train_model()
    val_loss, val_acc = evaluate_model()
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
