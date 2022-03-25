# review vital

<img src="https://user-images.githubusercontent.com/58410530/151568274-81e7783c-7b94-4a25-b8d4-42b98771553c.png">

## setup
```sh
$ git clone https://github.com/kakubin/review_vital
$ cd review_vital && make && make run
```

## run
```
# 1. pure bert
python ./pure_bert.py

# 2. use cnn to get features from the entire review
# make embeddings of sentences
python ./embedding.py
# train transformer and save it
python ./cnn_train.py

# 3. use transformer to get features from the entire review
# make embeddings of sentences
python ./embedding.py
# train transformer and save it
python ./bert_train.py
```

## data source
All Review Data attributes to Amazon.com, inc.
Downloaded [here](https://nijianmo.github.io/amazon/index.html)
