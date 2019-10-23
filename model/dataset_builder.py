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
        test_items = reader.items[0:test_count]
        logger.info('train item size: {0}'.format(len(train_items)))
        logger.info('test item size: {0}'.format(len(test_items)))

        train_dataset_size = 0
        test_dataset_size = 0
        if self.reader.infer_method:
            train_dataset_size += len(train_items)
            test_dataset_size += len(test_items)
        if self.reader.infer_variable:
            for item in train_items:
                train_dataset_size += len(self._filter_variable_aliases(item.aliases))
            for item in test_items:
                test_dataset_size += len(self._filter_variable_aliases(item.aliases))

        logger.info('train dataset size: {0}'.format(train_dataset_size))
        logger.info('test dataset size: {0}'.format(test_dataset_size))

        self.train_items = train_items
        self.test_items = test_items
        self.train_dataset = None
        self.test_dataset = None

        logger.info('OOV rate: {0}'.format(self._out_of_vocabulary_rate(option, reader, train_items, test_items)))

    def _filter_variable_aliases(self, aliases):
        return [alias_name for alias_name in aliases if alias_name.startswith("@var_")]

    def refresh_train_dataset(self):
        """refresh training dataset (shuffling path contexts and picking up items (#items <= max_path_length)"""
        inputs_id, inputs_starts, inputs_paths, inputs_ends, inputs_label = self.build_data(self.reader, self.train_items, self.option.max_path_length)
        self.train_dataset = CodeDataset(inputs_id, inputs_starts, inputs_paths, inputs_ends, inputs_label)

    def refresh_test_dataset(self):
        """refresh test dataset (shuffling path contexts and picking up items (#items <= max_path_length)"""
        inputs_id, inputs_starts, inputs_paths, inputs_ends, inputs_label = self.build_data(self.reader, self.test_items, self.option.max_path_length)
        self.test_dataset = CodeDataset(inputs_id, inputs_starts, inputs_paths, inputs_ends, inputs_label)

    def _get_labels(self, option, reader, normalized_label):
        if option.eval_method == 'exact':
            return [normalized_label]
        else:
            label_index = reader.label_vocab.stoi[normalized_label]
            subtokens = reader.label_vocab.itosubtokens[label_index]
            return subtokens

    def _out_of_vocabulary_rate(self, option, reader, train_items, test_items):
        train_vocab = set()

        tokens_match = 0
        tokens_count = 0
        if self.reader.infer_method:
            for item in train_items:
                tokens = self._get_labels(option, reader, item.normalized_label)
                for token in tokens:
                    train_vocab.add(token)

        if self.reader.infer_variable:
            for item in train_items:
                alias_names = self._filter_variable_aliases(item.aliases)
                alias_indexes = [reader.terminal_vocab.stoi[alias_name] for alias_name in alias_names]

                for alias_name, var_token_index in zip(alias_names, alias_indexes):
                    normalized_var_name = item.aliases[alias_name]
                    tokens = self._get_labels(option, reader, normalized_var_name)
                    for token in tokens:
                        train_vocab.add(token)

        if self.reader.infer_method:
            for item in test_items:
                tokens = self._get_labels(option, reader, item.normalized_label)
                tokens_match += len([token for token in tokens if token in train_vocab])
                tokens_count += len(tokens)

        if self.reader.infer_variable:
            for item in test_items:
                alias_names = self._filter_variable_aliases(item.aliases)
                alias_indexes = [reader.terminal_vocab.stoi[alias_name] for alias_name in alias_names]

                for alias_name, var_token_index in zip(alias_names, alias_indexes):
                    normalized_var_name = item.aliases[alias_name]
                    tokens = self._get_labels(option, reader, normalized_var_name)
                    tokens_match += len([token for token in tokens if token in train_vocab])
                    tokens_count += len(tokens)
        return 1.0 - tokens_match / tokens_count

    def build_data(self, reader, items, max_path_length):
        inputs_id = []
        inputs_starts = []
        inputs_paths = []
        inputs_ends = []
        inputs_label = []
        label_vocab_stoi = reader.label_vocab.stoi
        terminal_vocab_stoi = reader.terminal_vocab.stoi
        question_token_index = reader.QUESTION_TOKEN_INDEX

        if self.reader.infer_method:
            # replace @method_0 with @question
            method_token_index = terminal_vocab_stoi["@method_0"]

            for item in items:
                inputs_id.append(item.id)
                label_index = label_vocab_stoi[item.normalized_label]
                inputs_label.append(label_index)
                starts = []
                paths = []
                ends = []

                random.shuffle(item.path_contexts)
                for start, path, end in item.path_contexts[:max_path_length]:
                    if start == method_token_index:
                        start = question_token_index
                    starts.append(start)

                    paths.append(path)

                    if end == method_token_index:
                        end = question_token_index
                    ends.append(end)
                starts = self.pad_inputs(starts, max_path_length)
                paths = self.pad_inputs(paths, max_path_length)
                ends = self.pad_inputs(ends, max_path_length)
                inputs_starts.append(starts)
                inputs_paths.append(paths)
                inputs_ends.append(ends)

        if self.reader.infer_variable:
            # replace @var_XX with @question
            variable_indexes = self.reader.variable_indexes
            new_variable_indexes = [*variable_indexes]
            new_var_dict = {k: v for k, v in zip(variable_indexes, new_variable_indexes)}

            for item in items:
                alias_names = self._filter_variable_aliases(item.aliases)
                alias_indexes = [terminal_vocab_stoi[alias_name] for alias_name in alias_names]

                if self.reader.shuffle_variable_indexes:
                    random.shuffle(new_variable_indexes)
                    new_var_dict = {k: v for k, v in zip(variable_indexes, new_variable_indexes)}

                # filter on path-contexts related to variables of interest
                var_path_contexts = [pc for pc in item.path_contexts
                                     if pc[0] in alias_indexes or pc[2] in alias_indexes]
                random.shuffle(var_path_contexts)

                for alias_name, var_token_index in zip(alias_names, alias_indexes):
                    normalized_var_name = item.aliases[alias_name]
                    label_index = label_vocab_stoi[normalized_var_name]

                    inputs_id.append(item.id)
                    inputs_label.append(label_index)
                    starts = []
                    paths = []
                    ends = []

                    for start, path, end in [pc for pc in var_path_contexts
                                             if pc[0] == var_token_index or pc[2] == var_token_index]:
                        if start == var_token_index:
                            start = question_token_index
                        else:
                            start = new_var_dict.get(start, start)
                        starts.append(start)

                        paths.append(path)

                        if end == var_token_index:
                            end = question_token_index
                        else:
                            end = new_var_dict.get(end, end)
                        ends.append(end)
                    starts = starts[:max_path_length]
                    paths = paths[:max_path_length]
                    ends = ends[:max_path_length]
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
