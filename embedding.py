import torch
import nltk
from transformers import AutoTokenizer, AutoModel
from utils import ReviewData, embedding
import pickle
from datetime import datetime

nltk.download('punkt')


def all_review_embedding(review_text, tokenizer, model):
    sentence_embeddings = []
    cls = torch.zeros([1, 768])
    cls[0][0] = 1
    sentence_embeddings.append(cls)

    for sentence in nltk.tokenize.sent_tokenize(review_text):
        outputs = embedding(sentence, tokenizer, model, device)
        outputs = outputs.last_hidden_state.detach()[0][0]
        sentence_embeddings.append(outputs.unsqueeze(0).to('cpu'))

    eos = torch.zeros([1, 768])
    cls[0][1] = 1
    sentence_embeddings.append(eos)

    padding_len = 128 - len(sentence_embeddings)
    padding = torch.zeros([padding_len, 768])
    return torch.cat([*sentence_embeddings, padding])


def main(mode, options):
    df = ReviewData.fetch('All_Beauty', **options)

    embeddings = []
    for review_text in df['reviewText']:
        embeddings.append(all_review_embedding(review_text, tokenizer, model).unsqueeze(0))

    embeddings = torch.cat(embeddings)

    VOTE_THRESHOLD = 1
    targets = [[0., 1.] if vote >= VOTE_THRESHOLD else [1., 0.] for vote in df['vote']]
    targets = torch.tensor(targets)

    now = datetime.now().strftime("%Y%m%d%H%M%S")
    with open(f'pickle/{mode}/embedding_{now}.pickle', 'wb') as f:
        data = {
                'embeddings': embeddings,
                'targets': targets
                }
        pickle.dump(data, f)


device = "cuda:1" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

for mode in ['train', 'test']:
    print(mode)
    options = {
            'size': 5000,
            'threshold': 1,
            'ratio': 0.5
            }

    main(mode, options)
