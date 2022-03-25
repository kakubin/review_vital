import copy
import torch
from torch import nn
import time
import math
from models import TransformerModel
from utils import load_embedding_data, batchfy, make_padding_mask
from sklearn.metrics import (
    classification_report,
    accuracy_score,
)


def train(model: nn.Module, embeddings, targets) -> None:
    model.train()

    for batch_embeddings, batch_targets in batchfy(embeddings, targets, BATCH_SIZE, device):
        padding_mask = make_padding_mask(batch_embeddings).to(device)

        outputs = model(batch_embeddings, padding_mask)
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
            padding_mask = make_padding_mask(batch_embeddings).to(device)
            outputs = model(batch_embeddings, padding_mask)
            total_loss += BATCH_SIZE * criterion(outputs, batch_targets).item()
    return total_loss / embeddings.shape[0]


BATCH_SIZE = 1
EPOCH_SIZE = 10

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = TransformerModel(seq_len=128, d_model=768).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
        model.parameters(),
        lr=2e-3
        )
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

embeddings, targets = load_embedding_data('train')
best_val_loss = float('inf')

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


embeddings, targets = load_embedding_data('test')

batch_size = 1

predict = []
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

for batch_embeddings, _ in batchfy(embeddings, targets, device=device):
    padding_mask = make_padding_mask(batch_embeddings).to(device)

    with torch.no_grad():
        outputs = model(batch_embeddings, padding_mask).to('cpu').detach()
        outputs = [[0., 1.] if row[0] < row[1] else [1., 0.] for row in outputs]
        predict += outputs

predict = torch.tensor(predict)

batch_amount = targets.shape[0] // batch_size
batched_seq_len = batch_amount * batch_size
print("[ Report ]\n", classification_report(targets[:batched_seq_len], predict))
print("[ Accuracy ]", accuracy_score(targets[:batched_seq_len], predict))
