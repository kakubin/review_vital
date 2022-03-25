"""Entry point for all things utils."""

from .review_data import ReviewData
import pickle
import torch
from os import listdir, path, getcwd


def embedding(sentence, tokenizer, model, device=None):
    inputs = tokenizer(
            sentence,
            padding='max_length',
            max_length=512,
            truncation=True,
            return_tensors="pt"
            ).to(device)
    if device:
        model = model.to(device)
    return model(**inputs)


def load_embedding_data(mode):
    pickle_dir = path.join(getcwd(), f"pickle/{mode}")
    pickle_files = [f for f in listdir(pickle_dir) if path.isfile(path.join(pickle_dir, f))]
    latest_file = sorted(pickle_files)[-1]

    print(f'loading picklefile: pickle/{mode}/{latest_file}')

    with open(f'./pickle/{mode}/{latest_file}', 'rb') as f:
        data = pickle.load(f)
        embeddings = data['embeddings']
        targets = data['targets']
    return embeddings, targets


def batchfy(embeddings: torch.tensor, targets: torch.tensor, bsz=1, device='cpu'):
    batch_amount = embeddings.size(0) // bsz
    embeddings = embeddings[:batch_amount * bsz]
    embeddings_shape = embeddings.shape
    embeddings = torch.transpose(embeddings.view(bsz, batch_amount, *embeddings_shape[1:]), 0, 1).contiguous()
    targets = targets[:batch_amount * bsz]
    targets_shape = targets.shape
    targets = torch.transpose(targets.view(bsz, batch_amount, *targets_shape[1:]), 0, 1).contiguous()
    for i in range(batch_amount):
        yield embeddings[i].to(device), targets[i].to(device)


def make_padding_mask(embeddings):
    padding_mask = []
    for review in embeddings:
        review_padding = [(sentence_embedding.sum() == torch.tensor(0.)) for sentence_embedding in review]
        padding_mask.append(review_padding)
    return torch.tensor(padding_mask)
