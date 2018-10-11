# -*- coding: utf8 -*-

__author__ = "Isao Sonobe"
__copyright__ = "Copyright (C) 2018 Isao Sonobe"

import torch

from model.dataset_reader import *

import random
import logging

logger = logging.getLogger()


class DatasetBuilder(object):
    """transform dataset for training and test"""

    def __init__(self, reader, option, split_ratio=0.2):
        self.reader = reader
        self.option = option

        test_count = int(len(reader.items) * split_ratio)
        random.shuffle(reader.items)
        train_items = reader.items[test_count:]
        test_items = reader.items[0: test_count]
        logger.info('train dataset size: {0}'.format(len(train_items)))
        logger.info('test dataset size: {0}'.format(len(test_items)))

        self.train_items = train_items
        self.test_items = test_items

    def refresh_train_dataset(self):
        """refresh training dataset (shuffling path contexts and picking up items (#items <= max_path_length)"""
        inputs_id, inputs_starts, inputs_paths, inputs_ends, inputs_label = self.build_data(self.reader, self.train_items, self.option.max_path_length)
        self.train_dataset = CodeDataset(inputs_id, inputs_starts, inputs_paths, inputs_ends, inputs_label)

    def refresh_test_dataset(self):
        """refresh test dataset (shuffling path contexts and picking up items (#items <= max_path_length)"""
        inputs_id, inputs_starts, inputs_paths, inputs_ends, inputs_label = self.build_data(self.reader, self.test_items, self.option.max_path_length)
        self.test_dataset = CodeDataset(inputs_id, inputs_starts, inputs_paths, inputs_ends, inputs_label)

    def build_data(self, reader, items, max_path_length):
        inputs_id = []
        inputs_starts = []
        inputs_paths = []
        inputs_ends = []
        inputs_label = []
        label_vocab_stoi = reader.label_vocab.stoi
        for item in items:
            inputs_id.append(item.id)
            label_index = label_vocab_stoi[item.normalized_label]
            inputs_label.append(label_index)
            starts = []
            paths = []
            ends = []

            random.shuffle(item.path_contexts)
            for start, path, end in item.path_contexts[:max_path_length]:
                starts.append(start)
                paths.append(path)
                ends.append(end)
            starts = self.pad_inputs(starts, max_path_length)
            paths = self.pad_inputs(paths, max_path_length)
            ends = self.pad_inputs(ends, max_path_length)
            inputs_starts.append(starts)
            inputs_paths.append(paths)
            inputs_ends.append(ends)
        inputs_starts = torch.tensor(inputs_starts, dtype=torch.long)
        inputs_paths = torch.tensor(inputs_paths, dtype=torch.long)
        inputs_ends = torch.tensor(inputs_ends, dtype=torch.long)
        inputs_label = torch.tensor(inputs_label, dtype=torch.long)
        return inputs_id, inputs_starts, inputs_paths, inputs_ends, inputs_label

    def pad_inputs(self, data, length, pad_value=0):
        """pad values"""

        assert len(data) <= length

        count = length - len(data)
        data.extend([pad_value] * count)
        return data
