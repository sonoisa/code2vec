# -*- coding: utf8 -*-

__author__ = "Isao Sonobe"
__copyright__ = "Copyright (C) 2018 Isao Sonobe"

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math

NINF = - 3.4 * math.pow(10, 38)  # -Inf


class Code2Vec(nn.Module):
    """the code2vec model"""

    def __init__(self, option):
        super(Code2Vec, self).__init__()
        self.option = option
        self.terminal_embedding = nn.Embedding(option.terminal_count, option.terminal_embed_size)
        self.path_embedding = nn.Embedding(option.path_count, option.path_embed_size)
        self.input_linear = nn.Linear(option.terminal_embed_size * 2 + option.path_embed_size, option.encode_size, bias=False)
        self.input_layer_norm = nn.LayerNorm(option.encode_size)

        if 0.0 < option.dropout_prob < 1.0:
            self.input_dropout = nn.Dropout(p=option.dropout_prob)
        else:
            self.input_dropout = None

        self.attention_parameter = Parameter(torch.nn.init.xavier_normal_(torch.zeros(option.encode_size, 1, dtype=torch.float32, requires_grad=True)).view(-1), requires_grad=True)

        if option.angular_margin_loss:
            self.output_linear = Parameter(torch.FloatTensor(option.label_count, option.encode_size))
            nn.init.xavier_uniform_(self.output_linear)
            self.cos_m = math.cos(option.angular_margin)
            self.sin_m = math.sin(option.angular_margin)
            self.th = math.cos(math.pi - option.angular_margin)
            self.mm = math.sin(math.pi - option.angular_margin) * option.angular_margin
        else:
            self.output_linear = nn.Linear(option.encode_size, option.label_count, bias=True)
            self.output_linear.bias.data.fill_(0.0)

    def forward(self, starts, paths, ends, label):
        option = self.option

        # embedding
        embed_starts = self.terminal_embedding(starts)
        embed_paths = self.path_embedding(paths)
        embed_ends = self.terminal_embedding(ends)
        combined_context_vectors = torch.cat((embed_starts, embed_paths, embed_ends), dim=2)

        # FNN, Layer Normalization, tanh
        combined_context_vectors = self.input_linear(combined_context_vectors)
        ccv_size = combined_context_vectors.size()
        combined_context_vectors = self.input_layer_norm(combined_context_vectors.view(-1, option.encode_size)).view(ccv_size)
        combined_context_vectors = torch.tanh(combined_context_vectors)

        # dropout
        if self.input_dropout is not None:
            combined_context_vectors = self.input_dropout(combined_context_vectors)

        # attention
        attn_mask = (starts > 0).float()
        attention = self.get_attention(combined_context_vectors, attn_mask)

        # code vector
        expanded_attn = attention.unsqueeze(-1).expand_as(combined_context_vectors)
        code_vector = torch.sum(torch.mul(combined_context_vectors, expanded_attn), dim=1)

        if option.angular_margin_loss:
            # angular margin loss
            cosine = F.linear(F.normalize(code_vector), F.normalize(self.output_linear))
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
            phi = cosine * self.cos_m - sine * self.sin_m
            phi = torch.where(cosine > 0, phi, cosine)
            one_hot = torch.zeros(cosine.size(), device=option.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            outputs = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            outputs *= option.inverse_temp
        else:
            # FNN
            outputs = self.output_linear(code_vector)

        # if opt.training and opt.dropout_prob < 1.0:
        #     outputs = F.dropout(outputs, p=opt.dropout_prob, training=opt.training)

        return outputs, code_vector, attention

    def get_attention(self, vectors, mask):
        """calculate the attention of the (masked) context vetors. mask=1: meaningful value, mask=0: padded."""
        expanded_attn_param = self.attention_parameter.unsqueeze(0).expand_as(vectors)
        attn_ca = torch.mul(torch.sum(vectors * expanded_attn_param, dim=2), mask) + (1 - mask) * NINF
        # attn_ca = torch.sum(vectors * expanded_attn_param, dim=2)
        # attn_ca[mask == 0] = NINF
        attention = F.softmax(attn_ca, dim=1)

        # expanded_attn_param = self.attention_parameter.unsqueeze(0).expand_as(vectors)
        # attn_ca = torch.mul(torch.sum(vectors * expanded_attn_param, dim=2), mask)
        # attn_max, _ = torch.max(attn_ca, dim=1, keepdim=True)
        # attn_exp = torch.mul(torch.exp(attn_ca - attn_max), mask)
        # attn_sum = torch.sum(attn_exp, dim=1, keepdim=True)
        # attention = torch.div(attn_exp, attn_sum.expand_as(attn_exp) + eps)

        return attention

