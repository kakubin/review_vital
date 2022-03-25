import copy
import torch
from torch import nn
from models import Classification
import time
import math
from utils import batchfy, embedding
from utils import ReviewData
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    )


def train(model: nn.Module, embeddings, targets) -> None:
    model.train()
    for batch_embeddings, batch_targets in batchfy(embeddings, targets, BATCH_SIZE, device):
        outputs = model(batch_embeddings)
        loss = criterion(outputs, batch_targets)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()


def evaluate(model, embeddings, targets) -> float:
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch_embeddings, batch_targets in batchfy(embeddings, targets, BATCH_SIZE, device):

            outputs = model(batch_embeddings)
            total_loss += BATCH_SIZE * criterion(outputs, batch_targets).item()
        return total_loss / embeddings.shape[0]


BATCH_SIZE = 1
EPOCH_SIZE = 50
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = Classification(d_model=768, output_dim=2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

best_val_loss = float('inf')
best_model = None

options = {
        'size': 5000,
        'threshold': 1,
        'ratio': 0.5
        }
df = ReviewData.fetch('All_Beauty', **options)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
extractor = AutoModel.from_pretrained("bert-base-uncased")

embeddings = []
for row in df['reviewText']:
    outputs = embedding(row, tokenizer, extractor, device)
    outputs = outputs.last_hidden_state.to('cpu').detach().numpy().copy()
    embeddings.append(outputs[0][0])

embeddings = torch.tensor(embeddings)

VOTE_THRESHOLD = 1
targets = df['vote'].apply(lambda x: x >= VOTE_THRESHOLD)
targets = [[0., 1.] if x else [1., 0.] for x in targets]
targets = torch.tensor(targets)

for epoch in range(1, EPOCH_SIZE+1):
    epoch_start_time = time.time()
    train(model, embeddings, targets)
    val_loss = evaluate(model, embeddings, targets)
    val_ppl = math.exp(val_loss)
    elapsed = time.time() - epoch_start_time
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)

    scheduler.step()

# prediction
df = ReviewData.fetch('All_Beauty', **options)

embeddings = []
for row in df['reviewText']:
    outputs = embedding(row, tokenizer, extractor, device)
    outputs = outputs.last_hidden_state.to('cpu').detach().numpy().copy()
    embeddings.append(outputs[0][0])

embeddings = torch.tensor(embeddings)

VOTE_THRESHOLD = 1
targets = df['vote'].apply(lambda x: x >= VOTE_THRESHOLD)
targets = [[0., 1.] if x else [1., 0.] for x in targets]
targets = torch.tensor(targets)

predict = []
for batch_embeddings, _ in batchfy(embeddings, targets, BATCH_SIZE, device):
    with torch.no_grad():
        outputs = model(batch_embeddings).to('cpu').detach()
        outputs = [[0., 1.] if row[0] < row[1] else [1., 0.] for row in outputs]
        predict += outputs

predict = torch.tensor(predict)

print("[ Report ]\n", classification_report(targets, predict))
print("[ Accuracy ]", accuracy_score(targets, predict))
