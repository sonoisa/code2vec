# -*- coding: utf8 -*-

__author__ = "Isao Sonobe"
__copyright__ = "Copyright (C) 2018 Isao Sonobe"

from .dataset import *

import logging
logger = logging.getLogger()


class VocabReader(object):
    """語彙をファイルから読み込む"""

    def __init__(self, filename):
        self.filename = filename

    def read(self):
        vocab = Vocab()
        with open(self.filename, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip(' \r\n\t').split('\t')
                index = int(data[0])
                if len(data) > 1:
                    name = data[1]
                else:
                    name = ""
                vocab.append(name, index)
        return vocab


class DatasetReader(object):
    """データセットをファイルから読み込む"""

    def __init__(self, corpus_path, path_index_path, terminal_index_path):
        self.path_vocab = VocabReader(path_index_path).read()
        logger.info('path vocab size: {0}'.format(self.path_vocab.len()))

        self.terminal_vocab = VocabReader(terminal_index_path).read()
        logger.info('terminal vocab size: {0}'.format(self.terminal_vocab.len()))

        self.label_vocab = Vocab()
        self.items = []
        self.load(corpus_path)

        logger.info('label vocab size: {0}'.format(self.label_vocab.len()))
        logger.info('corpus: {0}'.format(len(self.items)))

    def load(self, corpus_path):
        with open(corpus_path, mode="r", encoding="utf-8") as f:
            code_data = None
            path_contexts_append = None
            parse_mode = 0
            label_vocab = self.label_vocab
            label_vocab_append = label_vocab.append
            for line in f.readlines():
                line = line.strip(' \r\n\t')

                if line == '':
                    if code_data is not None:
                        self.items.append(code_data)
                        code_data = None
                    continue

                if code_data is None:
                    code_data = CodeData()
                    path_contexts_append = code_data.path_contexts.append

                if line.startswith('#'):
                    code_data.id = int(line[1:])
                elif line.startswith('label:'):
                    label = line[6:]
                    code_data.label = label
                    normalized_label = Vocab.normalize_method_name(label)
                    subtokens = Vocab.get_method_subtokens(normalized_label)
                    normalized_lower_label = normalized_label.lower()
                    code_data.normalized_label = normalized_lower_label
                    label_vocab_append(normalized_lower_label, subtokens=subtokens)
                elif line.startswith('class:'):
                    code_data.source = line[6:]
                elif line.startswith('paths:'):
                    parse_mode = 1
                elif line.startswith('vars:'):
                    parse_mode = 2
                elif line.startswith('doc:'):
                    doc = line[4:]
                elif parse_mode == 1:
                    path_context = line.split('\t')
                    path_contexts_append((int(path_context[0]), int(path_context[1]), int(path_context[2])))
                elif parse_mode == 2:
                    alias = line.split('\t')
                    code_data.aliases[alias[1]] = alias[0]

            if code_data is not None:
                self.items.append(code_data)
