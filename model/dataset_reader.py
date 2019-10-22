# -*- coding: utf8 -*-

__author__ = "Isao Sonobe"
__copyright__ = "Copyright (C) 2018 Isao Sonobe"

from .dataset import *

import logging
logger = logging.getLogger()

QUESTION_TOKEN_INDEX = 1
QUESTION_TOKEN_NAME = "@question"


class VocabReader(object):
    """read vocabulary file"""

    def __init__(self, filename, extra_tokens=[]):
        self.filename = filename
        self.extra_tokens = extra_tokens

    def read(self):
        vocab = Vocab()
        extra_size = len(self.extra_tokens)
        index = 1
        for name in self.extra_tokens:
            vocab.append(name, index)

        with open(self.filename, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip(' \r\n\t').split('\t')
                index = int(data[0])
                if index > 0:
                    index += extra_size
                if len(data) > 1:
                    name = data[1]
                else:
                    name = ""
                vocab.append(name, index)
        return vocab


class DatasetReader(object):
    """read dataset file"""

    def __init__(self, corpus_path, path_index_path, terminal_index_path, infer_method, infer_variable, shuffle_variable_indexes):
        self.path_vocab = VocabReader(path_index_path).read()
        logger.info('path vocab size: {0}'.format(self.path_vocab.len()))

        self.terminal_vocab = VocabReader(terminal_index_path, extra_tokens=[QUESTION_TOKEN_NAME]).read()
        logger.info('terminal vocab size: {0}'.format(self.terminal_vocab.len()))

        terminal_vocab_stoi = self.terminal_vocab.stoi
        self.variable_indexes = [terminal_vocab_stoi[term] for term in terminal_vocab_stoi if term.startswith("@var_")]
        logger.info('variable index size: {0}'.format(len(self.variable_indexes)))

        self.shuffle_variable_indexes = shuffle_variable_indexes
        self.QUESTION_TOKEN_NAME = QUESTION_TOKEN_NAME
        self.QUESTION_TOKEN_INDEX = QUESTION_TOKEN_INDEX
        self.infer_method = infer_method
        self.infer_variable = infer_variable
        logger.info("infer method names: {}, infer variable names: {}".format(self.infer_method, self.infer_variable))

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
                    if self.infer_method:
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
                    path_contexts_append((int(path_context[0]) + QUESTION_TOKEN_INDEX,
                                          int(path_context[1]),
                                          int(path_context[2]) + QUESTION_TOKEN_INDEX))
                elif parse_mode == 2:
                    alias = line.split('\t')
                    alias_name = alias[1]
                    original_name = alias[0]
                    normalized_var_name = Vocab.normalize_method_name(original_name)
                    subtokens = Vocab.get_method_subtokens(normalized_var_name)
                    normalized_lower_var_name = normalized_var_name.lower()
                    code_data.aliases[alias_name] = normalized_lower_var_name
                    if self.infer_variable and alias_name.startswith("@var_"):
                        label_vocab_append(normalized_lower_var_name, subtokens=subtokens)

            if code_data is not None:
                self.items.append(code_data)
