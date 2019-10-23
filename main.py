# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = "Isao Sonobe"
__copyright__ = "Copyright (C) 2018 Isao Sonobe"

import sys

import argparse
import numpy as np
import torch

from os import path

from distutils.util import strtobool

from torch.utils.data import DataLoader

from model.model import *
from model.dataset_builder import *

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

sys.path.append('.')


logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: %(message)s', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=123, help="random_seed")

parser.add_argument('--corpus_path', type=str, default="./dataset/corpus.txt", help="corpus_path")
parser.add_argument('--path_idx_path', type=str, default="./dataset/path_idxs.txt", help="path_idx_path")
parser.add_argument('--terminal_idx_path', type=str, default="./dataset/terminal_idxs.txt", help="terminal_idx_path")

parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
parser.add_argument('--terminal_embed_size', type=int, default=100, help="terminal_embed_size")
parser.add_argument('--path_embed_size', type=int, default=100, help="path_embed_size")
parser.add_argument('--encode_size', type=int, default=300, help="encode_size")
parser.add_argument('--max_path_length', type=int, default=200, help="max_path_length")

parser.add_argument('--model_path', type=str, default="./output", help="model_path")
parser.add_argument('--vectors_path', type=str, default="./output/code.vec", help="vectors_path")
parser.add_argument('--test_result_path', type=str, default=None, help="test_result_path")

parser.add_argument("--max_epoch", type=int, default=40, help="max_epoch")
parser.add_argument('--lr', type=float, default=0.01, help="lr")
parser.add_argument('--beta_min', type=float, default=0.9, help="beta_min")
parser.add_argument('--beta_max', type=float, default=0.999, help="beta_max")
parser.add_argument('--weight_decay', type=float, default=0.0, help="weight_decay")

parser.add_argument('--dropout_prob', type=float, default=0.25, help="dropout_prob")

parser.add_argument("--no_cuda", action="store_true", default=False, help="no_cuda")
parser.add_argument("--gpu", type=str, default="cuda:0", help="gpu")
parser.add_argument("--num_workers", type=int, default=4, help="num_workers")

parser.add_argument("--env", type=str, default=None, help="env")
parser.add_argument("--print_sample_cycle", type=int, default=10, help="print_sample_cycle")
parser.add_argument("--eval_method", type=str, default="subtoken", help="eval_method")

parser.add_argument("--find_hyperparams", action="store_true", default=False, help="find optimal hyperparameters")
parser.add_argument("--num_trials", type=int, default=100, help="num_trials")

parser.add_argument("--angular_margin_loss", action="store_true", default=False, help="use angular margin loss")
parser.add_argument("--angular_margin", type=float, default=0.5, help="angular margin")
parser.add_argument("--inverse_temp", type=float, default=30.0, help="inverse temperature")

parser.add_argument("--infer_method_name", type=lambda b: bool(strtobool(b)), default=True, help="infer method name like code2vec task")
parser.add_argument("--infer_variable_name", type=lambda b: bool(strtobool(b)), default=False, help="infer variable name like context2name task")
parser.add_argument("--shuffle_variable_indexes", type=lambda b: bool(strtobool(b)), default=False, help="shuffle variable indexes in the variable name inference task")

args = parser.parse_args()

device = torch.device(args.gpu if not args.no_cuda and torch.cuda.is_available() else "cpu")
logger.info("device: {0}".format(device))


if args.env == "tensorboard":
    from tensorboardX import SummaryWriter

if args.find_hyperparams:
    import optuna

class Option(object):
    """configurations of the model"""

    def __init__(self, reader):
        self.max_path_length = args.max_path_length

        self.terminal_count = reader.terminal_vocab.len()
        self.path_count = reader.path_vocab.len()
        self.label_count = reader.label_vocab.len()

        self.terminal_embed_size = args.terminal_embed_size
        self.path_embed_size = args.path_embed_size
        self.encode_size = args.encode_size

        self.dropout_prob = args.dropout_prob
        self.batch_size = args.batch_size
        self.eval_method = args.eval_method

        self.angular_margin_loss = args.angular_margin_loss
        self.angular_margin = args.angular_margin
        self.inverse_temp = args.inverse_temp

        self.device = device


def train():
    """train the model"""
    torch.manual_seed(args.random_seed)

    reader = DatasetReader(args.corpus_path, args.path_idx_path, args.terminal_idx_path,
                           infer_method=args.infer_method_name, infer_variable=args.infer_variable_name,
                           shuffle_variable_indexes=args.shuffle_variable_indexes)
    option = Option(reader)

    builder = DatasetBuilder(reader, option)

    label_freq = torch.tensor(reader.label_vocab.get_freq_list(), dtype=torch.float32).to(device)
    criterion = nn.NLLLoss(weight=1 / label_freq).to(device)

    model = Code2Vec(option).to(device)
    # print(model)
    # for param in model.parameters():
    #     print(type(param.data), param.size())

    learning_rate = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(args.beta_min, args.beta_max), weight_decay=args.weight_decay)

    _train(model, optimizer, criterion, option, reader, builder, None)


def _train(model, optimizer, criterion, option, reader, builder, trial):
    """train the model"""

    f1 = 0.0
    best_f1 = None
    last_loss = None
    last_accuracy = None
    bad_count = 0

    if args.env == "tensorboard":
        summary_writer = SummaryWriter()
    else:
        summary_writer = None

    try:
        for epoch in range(args.max_epoch):
            train_loss = 0.0

            builder.refresh_train_dataset()
            train_data_loader = DataLoader(builder.train_dataset, batch_size=option.batch_size, shuffle=True, num_workers=args.num_workers)

            model.train()
            for i_batch, sample_batched in enumerate(train_data_loader):
                starts = sample_batched['starts'].to(option.device)
                paths = sample_batched['paths'].to(option.device)
                ends = sample_batched['ends'].to(option.device)
                label = sample_batched['label'].to(device)

                optimizer.zero_grad()
                preds, _, _ = model.forward(starts, paths, ends, label)
                loss = calculate_loss(preds, label, criterion, option)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            builder.refresh_test_dataset()
            test_data_loader = DataLoader(builder.test_dataset, batch_size=option.batch_size, shuffle=True, num_workers=args.num_workers)
            test_loss, accuracy, precision, recall, f1 = test(model, test_data_loader, criterion, option, reader.label_vocab)

            if args.env == "floyd":
                print("epoch {0}".format(epoch))
                print('{{"metric": "train_loss", "value": {0}}}'.format(train_loss))
                print('{{"metric": "test_loss", "value": {0}}}'.format(test_loss))
                print('{{"metric": "accuracy", "value": {0}}}'.format(accuracy))
                print('{{"metric": "precision", "value": {0}}}'.format(precision))
                print('{{"metric": "recall", "value": {0}}}'.format(recall))
                print('{{"metric": "f1", "value": {0}}}'.format(f1))
            else:
                logger.info("epoch {0}".format(epoch))
                logger.info('{{"metric": "train_loss", "value": {0}}}'.format(train_loss))
                logger.info('{{"metric": "test_loss", "value": {0}}}'.format(test_loss))
                logger.info('{{"metric": "accuracy", "value": {0}}}'.format(accuracy))
                logger.info('{{"metric": "precision", "value": {0}}}'.format(precision))
                logger.info('{{"metric": "recall", "value": {0}}}'.format(recall))
                logger.info('{{"metric": "f1", "value": {0}}}'.format(f1))
            if args.env == "tensorboard":
                summary_writer.add_scalar('metric/train_loss', train_loss, epoch)
                summary_writer.add_scalar('metric/test_loss', test_loss, epoch)
                summary_writer.add_scalar('metric/accuracy', accuracy, epoch)
                summary_writer.add_scalar('metric/precision', precision, epoch)
                summary_writer.add_scalar('metric/recall', recall, epoch)
                summary_writer.add_scalar('metric/f1', f1, epoch)

            if trial is not None:
                intermediate_value = 1.0 - f1
                trial.report(intermediate_value, epoch)
                if trial.should_prune(epoch):
                    raise optuna.structs.TrialPruned()

            if epoch > 1 and epoch % args.print_sample_cycle == 0 and trial is None:
                print_sample(reader, model, test_data_loader, option)

            if best_f1 is None or best_f1 < f1:
                if args.env == "floyd":
                    print('{{"metric": "best_f1", "value": {0}}}'.format(f1))
                else:
                    logger.info('{{"metric": "best_f1", "value": {0}}}'.format(f1))
                if args.env == "tensorboard":
                    summary_writer.add_scalar('metric/best_f1', f1, epoch)

                best_f1 = f1
                if trial is None:
                    vector_file = args.vectors_path
                    with open(vector_file, "w") as f:
                        f.write("{0}\t{1}\n".format(len(reader.items), option.encode_size))
                    write_code_vectors(reader, model, train_data_loader, option, vector_file, "a", None)
                    write_code_vectors(reader, model, test_data_loader, option, vector_file, "a", args.test_result_path)
                    torch.save(model.state_dict(), path.join(args.model_path, "code2vec.model"))

            if last_loss is None or train_loss < last_loss or last_accuracy is None or last_accuracy < accuracy:
                last_loss = train_loss
                last_accuracy = accuracy
                bad_count = 0
            else:
                bad_count += 1
            if bad_count > 10:
                print('early stop loss:{0}, bad:{1}'.format(train_loss, bad_count))
                print_sample(reader, model, test_data_loader, option)
                break

    finally:
        if args.env == "tensorboard":
            summary_writer.close()

    return 1.0 - f1


def calculate_loss(predictions, label, criterion, option):
    # preds = F.log_softmax(predictions, dim=1)
    #
    # batch_size = predictions.size()[0]
    # y_onehot = torch.FloatTensor(batch_size, option.label_count).to(device)
    # y_onehot.zero_()
    # y_onehot.scatter_(1, label.view(-1, 1), 1)
    #
    # loss = -torch.mean(torch.sum(preds * y_onehot, dim=1))

    preds = F.log_softmax(predictions, dim=1)
    loss = criterion(preds, label)

    return loss


def test(model, data_loader, criterion, option, label_vocab):
    """test the model"""
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        expected_labels = []
        actual_labels = []

        for i_batch, sample_batched in enumerate(data_loader):
            starts = sample_batched['starts'].to(option.device)
            paths = sample_batched['paths'].to(option.device)
            ends = sample_batched['ends'].to(option.device)
            label = sample_batched['label'].to(device)
            expected_labels.extend(label)

            preds, _, _ = model.forward(starts, paths, ends, label)
            loss = calculate_loss(preds, label, criterion, option)
            test_loss += loss.item()
            _, preds_label = torch.max(preds, dim=1)
            actual_labels.extend(preds_label)

        expected_labels = np.array(expected_labels)
        actual_labels = np.array(actual_labels)
        accuracy, precision, recall, f1 = None, None, None, None
        if args.eval_method == 'exact':
            accuracy, precision, recall, f1 = exact_match(expected_labels, actual_labels)
        elif args.eval_method == 'subtoken':
            accuracy, precision, recall, f1 = subtoken_match(expected_labels, actual_labels, label_vocab)
        elif args.eval_method == 'ave_subtoken':
            accuracy, precision, recall, f1 = averaged_subtoken_match(expected_labels, actual_labels, label_vocab)
        return test_loss, accuracy, precision, recall, f1


def exact_match(expected_labels, actual_labels):
    expected_labels = np.array(expected_labels, dtype=np.uint64)
    actual_labels = np.array(actual_labels, dtype=np.uint64)
    precision, recall, f1, _ = precision_recall_fscore_support(expected_labels, actual_labels, average='weighted')
    accuracy = accuracy_score(expected_labels, actual_labels)
    return accuracy, precision, recall, f1


def averaged_subtoken_match(expected_labels, actual_labels, label_vocab):
    subtokens_accuracy = []
    subtokens_precision = []
    subtokens_recall = []
    subtokens_f1 = []
    for expected, actual in zip(expected_labels.tolist(), actual_labels.tolist()):
        exp_subtokens = label_vocab.itosubtokens[expected]
        actual_subtokens = label_vocab.itosubtokens[actual]
        match = 0
        for subtoken in exp_subtokens:
            if subtoken in actual_subtokens:
                match += 1
        acc = match / float(len(exp_subtokens) + len(actual_subtokens) - match)
        rec = match / float(len(exp_subtokens))
        prec = match / float(len(actual_subtokens))
        if prec + rec > 0:
            subtoken_f1 = 2.0 * prec * rec / (prec + rec)
        else:
            subtoken_f1 = 0.0
        subtokens_accuracy.append(acc)
        subtokens_precision.append(prec)
        subtokens_recall.append(rec)
        subtokens_f1.append(subtoken_f1)

    ave_accuracy = np.average(subtokens_accuracy)
    ave_precision = np.average(subtokens_precision)
    ave_recall = np.average(subtokens_recall)
    ave_f1 = np.average(subtokens_f1)
    return ave_accuracy, ave_precision, ave_recall, ave_f1


def subtoken_match(expected_labels, actual_labels, label_vocab):
    subtokens_match = 0.0
    subtokens_expected_count = 0.0
    subtokens_actual_count = 0.0
    for expected, actual in zip(expected_labels.tolist(), actual_labels.tolist()):
        exp_subtokens = label_vocab.itosubtokens[expected.item()]
        actual_subtokens = label_vocab.itosubtokens[actual.item()]
        for subtoken in exp_subtokens:
            if subtoken in actual_subtokens:
                subtokens_match += 1
        subtokens_expected_count += len(exp_subtokens)
        subtokens_actual_count += len(actual_subtokens)

    accuracy = subtokens_match / (subtokens_expected_count + subtokens_actual_count - subtokens_match)
    precision = subtokens_match / subtokens_actual_count
    recall = subtokens_match / subtokens_expected_count
    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return accuracy, precision, recall, f1


def print_sample(reader, model, data_loader, option):
    """print one data that leads correct prediction with the trained model"""
    model.eval()
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(data_loader):
            starts = sample_batched['starts'].to(option.device)
            paths = sample_batched['paths'].to(option.device)
            ends = sample_batched['ends'].to(option.device)
            label = sample_batched['label'].to(option.device)

            preds, code_vector, attn = model.forward(starts, paths, ends, label)
            _, preds_label = torch.max(preds, dim=1)

            for i in range(len(starts)):
                if preds_label[i] == label[i]:
                    # 予測と正解が一致したデータを1つだけ表示する。
                    start_names = [reader.terminal_vocab.itos[v.item()] for v in starts[i]]
                    path_names = [reader.path_vocab.itos[v.item()] for v in paths[i]]
                    end_names = [reader.terminal_vocab.itos[v.item()] for v in ends[i]]
                    label_name = reader.label_vocab.itos[label[i].item()]
                    pred_label_name = reader.label_vocab.itos[preds_label[i].item()]
                    attentions = attn.cpu()[i]

                    for start, path, end, attention in zip(start_names, path_names, end_names, attentions):
                        if start != "<PAD/>":
                            logger.info("{0} {1} {2} [{3}]".format(start, path, end, attention))
                    logger.info('expected label: {0}'.format(label_name))
                    logger.info('actual label:   {0}'.format(pred_label_name))
                    return


def write_code_vectors(reader, model, data_loader, option, vector_file, mode, test_result_file):
    """sav the code vectors"""
    model.eval()
    with torch.no_grad():
        if test_result_file is not None:
            fr = open(test_result_file, "w")
        else:
            fr = None

        with open(vector_file, mode) as fv:
            for i_batch, sample_batched in enumerate(data_loader):
                id = sample_batched['id']
                starts = sample_batched['starts'].to(option.device)
                paths = sample_batched['paths'].to(option.device)
                ends = sample_batched['ends'].to(option.device)
                label = sample_batched['label'].to(option.device)

                preds, code_vector, _ = model.forward(starts, paths, ends, label)
                preds_prob, preds_label = torch.max(preds, dim=1)

                for i in range(len(starts)):
                    label_name = reader.label_vocab.itos[label[i].item()]
                    vec = code_vector.cpu()[i]
                    fv.write(label_name + "\t" + " ".join([str(e.item()) for e in vec]) + "\n")

                    if test_result_file is not None:
                        pred_name = reader.label_vocab.itos[preds_label[i].item()]
                        fr.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(id[i].item(), label_name == pred_name, label_name, pred_name, preds_prob[i].item()))

        if test_result_file is not None:
            fr.close()


#
# for optuna
#
def find_optimal_hyperparams():
    """find optimal hyperparameters"""
    torch.manual_seed(args.random_seed)

    reader = DatasetReader(args.corpus_path, args.path_idx_path, args.terminal_idx_path,
                           infer_method=args.infer_method_name, infer_variable=args.infer_variable_name,
                           shuffle_variable_indexes=args.shuffle_variable_indexes)
    option = Option(reader)

    builder = DatasetBuilder(reader, option)

    label_freq = torch.tensor(reader.label_vocab.get_freq_list(), dtype=torch.float32).to(device)
    criterion = nn.NLLLoss(weight=1 / label_freq).to(device)

    def objective(trial):
        # option.max_path_length = int(trial.suggest_loguniform('max_path_length', 50, 200))
        # option.terminal_embed_size = int(trial.suggest_loguniform('terminal_embed_size', 50, 200))
        # option.path_embed_size = int(trial.suggest_loguniform('path_embed_size', 50, 200))
        option.encode_size = int(trial.suggest_loguniform('encode_size', 100, 300))
        option.dropout_prob = trial.suggest_loguniform('dropout_prob', 0.5, 0.9)
        option.batch_size = int(trial.suggest_loguniform('batch_size', 256, 2048))

        model = Code2Vec(option).to(device)
        # print(model)
        # for param in model.parameters():
        #     print(type(param.data), param.size())

        optimizer = get_optimizer(trial, model)

        return _train(model, optimizer, criterion, option, reader, builder, trial)

    study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=args.num_trials)

    best_params = study.best_params
    best_value = study.best_value
    if args.env == "floyd":
        print('best hyperparams: {0}'.format(best_params))
        print('best value: {0}'.format(best_value))
    else:
        logger.info("optimal hyperparams: {0}".format(best_params))
        logger.info('best value: {0}'.format(best_value))


def get_optimizer(trial, model):
    # optimizer = trial.suggest_categorical('optimizer', [adam, momentum])
    # weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
    # return optimizer(model, trial, weight_decay)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
    return adam(model, trial, weight_decay)


def adam(model, trial, weight_decay):
    lr = trial.suggest_loguniform('adam_lr', 1e-5, 1e-1)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def momentum(model, trial, weight_decay):
    lr = trial.suggest_loguniform('momentum_sgd_lr', 1e-5, 1e-1)
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)


#
# entry point
#
def main():
    if args.find_hyperparams:
        find_optimal_hyperparams()
    else:
        train()


if __name__ == '__main__':
    main()
