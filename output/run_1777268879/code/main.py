import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import math

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ===================== CONFIGURATION =====================
VOCAB_SIZE = 1000
MAX_LEN = 20
NUM_SAMPLES = 60
BATCH_SIZE = 16
EPOCHS = 3
HIDDEN_SIZE = 64
NUM_LAYERS = 2
NUM_HEADS = 4
INTERMEDIATE_SIZE = 256
NUM_CLASSES = 2
LEARNING_RATE = 2e-5

# ===================== SYNTHETIC DATASETS =====================
class SyntheticSentimentDataset(Dataset):
    def __init__(self, num_samples, vocab_size, max_len, seed=SEED):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.rng = np.random.RandomState(seed)
        # Generate synthetic data
        self.input_ids = []
        self.attention_masks = []
        self.labels = []
        for _ in range(num_samples):
            length = self.rng.randint(1, max_len + 1)
            tokens = self.rng.randint(0, vocab_size, size=length).tolist()
            # Pad to max_len
            padded = tokens + [0] * (max_len - length)
            mask = [1] * length + [0] * (max_len - length)
            label = self.rng.randint(0, 2)
            self.input_ids.append(padded)
            self.attention_masks.append(mask)
            self.labels.append(label)
        self.input_ids = torch.tensor(self.input_ids, dtype=torch.long)
        self.attention_masks = torch.tensor(self.attention_masks, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }

# Create datasets for IMDB, SST-2, Twitter (all synthetic)
datasets = {
    'IMDB': SyntheticSentimentDataset(NUM_SAMPLES, VOCAB_SIZE, MAX_LEN, seed=SEED),
    'SST-2': SyntheticSentimentDataset(NUM_SAMPLES, VOCAB_SIZE, MAX_LEN, seed=SEED+1),
    'Twitter': SyntheticSentimentDataset(NUM_SAMPLES, VOCAB_SIZE, MAX_LEN, seed=SEED+2)
}

# ===================== MINI TRANSFORMER MODELS =====================
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x: (batch, seq_len, hidden)
        attn_out, _ = self.attention(x, x, x, key_padding_mask=(mask == 0) if mask is not None else None)
        x = self.layernorm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.layernorm2(x + ffn_out)
        return x

class MiniBERT(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, intermediate_size, num_classes, max_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_len, hidden_size)
        self.embedding_dropout = nn.Dropout(0.1)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        self.pooler = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embedding_dropout(x)
        for layer in self.layers:
            x = layer(x, mask=attention_mask)
        # Use [CLS] token (first token) for classification
        cls_token = x[:, 0, :]
        pooled = self.pooler(cls_token)
        logits = self.classifier(pooled)
        return logits

# ===================== MODEL FACTORY =====================
def create_model(model_name):
    if model_name == 'BERT-base':
        return MiniBERT(VOCAB_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_HEADS, INTERMEDIATE_SIZE, NUM_CLASSES, MAX_LEN)
    elif model_name == 'RoBERTa-base':
        # Same architecture but with different initialization (simulated)
        model = MiniBERT(VOCAB_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_HEADS, INTERMEDIATE_SIZE, NUM_CLASSES, MAX_LEN)
        # Apply different weight scaling to simulate RoBERTa
        for param in model.parameters():
            if param.dim() > 1:
                nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
        return model
    elif model_name == 'DistilBERT-base':
        # Smaller version: fewer layers, smaller hidden
        model = MiniBERT(VOCAB_SIZE, HIDDEN_SIZE // 2, max(1, NUM_LAYERS - 1), max(1, NUM_HEADS // 2),
                         INTERMEDIATE_SIZE // 2, NUM_CLASSES, MAX_LEN)
        return model
    else:
        raise ValueError(f"Unknown model: {model_name}")

# ===================== TRAINING & EVALUATION =====================
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = correct / total
    # Compute precision, recall, F1
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    tp = np.sum((all_preds == 1) & (all_labels == 1))
    fp = np.sum((all_preds == 1) & (all_labels == 0))
    fn = np.sum((all_preds == 0) & (all_labels == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return accuracy, precision, recall, f1

# ===================== MAIN EXECUTION =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model_names = ['BERT-base', 'RoBERTa-base', 'DistilBERT-base']
dataset_names = ['IMDB', 'SST-2', 'Twitter']

results = {}

for model_name in model_names:
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print('='*50)
    model_results = {}
    for dataset_name in dataset_names:
        print(f"\n--- Dataset: {dataset_name} ---")
        dataset = datasets[dataset_name]
        # Split into train (80%) and eval (20%)
        train_size = int(0.8 * len(dataset))
        eval_size = len(dataset) - train_size
        train_dataset, eval_dataset = torch.utils.data.random_split(
            dataset, [train_size, eval_size],
            generator=torch.Generator().manual_seed(SEED)
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Create fresh model for each dataset
        model = create_model(model_name).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

        # Training loop
        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f}")

        # Evaluation
        accuracy, precision, recall, f1 = evaluate(model, eval_loader, device)
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        model_results[dataset_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    results[model_name] = model_results

# ===================== RESULTS TABLE =====================
print("\n\n" + "="*80)
print("FINAL RESULTS TABLE")
print("="*80)
header = f"{'Model':<20} {'Dataset':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}"
print(header)
print("-"*80)
for model_name in model_names:
    for dataset_name in dataset_names:
        metrics = results[model_name][dataset_name]
        row = f"{model_name:<20} {dataset_name:<12} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f}"
        print(row)
print("="*80)