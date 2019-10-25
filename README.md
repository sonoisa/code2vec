# code2vec
an implementation of "[code2vec: Learning Distributed Representations of Code](https://arxiv.org/abs/1803.09473)"


# Requirements
* python 3.6+
* pytorch 1.1+
* scikit-learn
* tensorboardX (optional)


# Usage

## train with "dataset"

```
python main.py --lr 0.01 --corpus_path ./dataset/corpus.txt --path_idx_path ./dataset/path_idxs.txt --terminal_idx_path ./dataset/terminal_idxs.txt --model_path ./output --vectors_path ./output/code.vec --terminal_embed_size 100 --path_embed_size 100 --encode_size 100 --max_epoch 40 --random_seed 1 --dropout_prob 0.25
```


## train with large "top11_dataset"

* top 11 dataset: http://groups.inf.ed.ac.uk/cup/codeattention/

concatenate dataset:

```
cat ./top11_dataset/splitted_corpus.* > ./top11_dataset/corpus.txt
```

train the model:

```
python main.py --batch_size 1024 --lr 0.01 --corpus_path ./top11_dataset/corpus.txt --path_idx_path ./top11_dataset/path_idxs.txt --terminal_idx_path ./top11_dataset/terminal_idxs.txt --model_path ./output --vectors_path ./output/code.vec --terminal_embed_size 100 --path_embed_size 100 --encode_size 100 --max_epoch 20 --random_seed 1 --dropout_prob 0.25
```


# License
MIT License
