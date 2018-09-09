import torch
from tensorboardX import SummaryWriter

vec_path = "./output/code.vec"

writer = SummaryWriter()

with open(vec_path, "r", encoding="utf-8") as f:
    labels = []
    vectors = []
    first = True
    for line in f.readlines():
        if first:
            first = False
            continue
        line = line.strip(' \r\n\t')
        data = line.split("\t")
        word = data[0]
        vector = [float(x) for x in data[1].split(" ")]
        labels.append(word)
        vectors.append(vector)

writer.add_embedding(torch.FloatTensor(vectors), metadata=labels)
