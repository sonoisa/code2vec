# -*- coding: utf8 -*-

__author__ = "Isao Sonobe"
__copyright__ = "Copyright (C) 2018 Isao Sonobe"

from torch.utils.data import Dataset

import logging
logger = logging.getLogger()


class CodeDataset(Dataset):
    """データセット（学習やテストの入力になるデータ）"""

    def __init__(self, ids, starts, paths, ends, labels, transform=None):
        self.ids = ids
        self.starts = starts
        self.paths = paths
        self.ends = ends
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        item = {
            'id': self.ids[index],
            'starts': self.starts[index],
            'paths': self.paths[index],
            'ends': self.ends[index],
            'label': self.labels[index]
        }
        if self.transform:
            item = self.transform(item)
        return item


class CodeData(object):
    """1つのメソッドに対応するデータ"""

    def __init__(self):
        self.id = None
        self.label = None
        self.path_contexts = []
        self.source = None
        self.aliases = {}


class Vocab(object):
    """終端記号やパス、ラベルの語彙"""

    def __init__(self):
        self.stoi = {}
        self.itos = {}

    def append(self, name, index=None):
        if name not in self.stoi:
            if index is None:
                index = len(self.stoi)
            self.stoi[name] = index
            self.itos[index] = name

    def len(self):
        return len(self.stoi)
