import gensim
import torch
from tensorboardX import SummaryWriter

vec_path = "./output/code.vec"

model = gensim.models.KeyedVectors.load_word2vec_format(vec_path, binary=True)
weights = model.vectors
labels = model.index2word

weights = weights[:1000]
labels = labels[:1000]

writer = SummaryWriter()
writer.add_embedding(torch.FloatTensor(weights), metadata=labels)
