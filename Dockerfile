FROM python:3.9.7-slim-bullseye

RUN pip install --upgrade pip

RUN pip3 install \
  numpy \
  pandas \
  scikit-learn \
  torch \
  transformers \
  sentencepiece \
  nltk

WORKDIR /review_vital
RUN python -c 'from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained("bert-base-uncased"); AutoModel.from_pretrained("bert-base-uncased")'
