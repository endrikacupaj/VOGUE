from __future__ import division
import os
import re
import time
import json
import nltk
import tqdm
import torch
import random
import hashlib
import logging
import sklearn
import numpy as np
import torch.nn as nn
from pathlib import Path
from args import get_parser

from constants import *

logger = logging.getLogger(__name__)

class NoamOpt:
    def __init__(self, optimizer, model_size=args.emb_dim, factor=args.factor, warmup=args.warmup):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class F1ScoreMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = []
        self.y_pred = []
        self.f1_score = 0

    def update(self, true, pred):
        self.y_true.append(true)
        self.y_pred.append(pred)
        self.f1_score = sklearn.metrics.f1_score(self.y_true, self.y_pred)

class Predictor(object):
    def __init__(self, model, vocabs, device):
        self.model = model
        self.vocabs = vocabs
        self.device = device

    def _prepare_input_tensor(self, input, vocab):
        while len(input) < args.max_input_size:
            input.append(PAD_TOKEN)

        tokenized_input = [START_TOKEN] + [t.lower() for t in input] + [CTX_TOKEN]
        numericalized = [vocab.stoi[token] if token in vocab.stoi else vocab.stoi[UNK_TOKEN] for token in tokenized_input]

        return torch.LongTensor(numericalized).unsqueeze(0).to(self.device)

    def predict(self, question, logical_form):
        self.model.eval()
        model_out = {}

        # prepare input
        question_tensor = self._prepare_input_tensor(question, self.vocabs[INPUT])

        with torch.no_grad():
            encoded_question = self.model.question_encoder(question_tensor)

            if not logical_form:
                dec_lf = [UNK_TOKEN for _ in range(args.max_input_size)]
                dec_lf_tensor = self._prepare_input_tensor(dec_lf, self.vocabs[LOGICAL_FORM])
                encoded_dec_lf = self.model.query_encoder(dec_lf_tensor)
            else:
                st_lf_tensor = self._prepare_input_tensor(logical_form, self.vocabs[LOGICAL_FORM])
                encoded_st_lf = self.model.query_encoder(st_lf_tensor)

                question_ctx = encoded_question[:, -1:, :]
                st_lf_ctx = encoded_st_lf[:, -1:, :]
                input_ctx = torch.cat([question_ctx, st_lf_ctx], dim=-1)
                st_pred = self.model.similarity_threshold(input_ctx).argmax(1).tolist()

                if self.vocabs[SIMILARITY_THRESHOLD].itos[st_pred[0]] == '1':
                    encoded_dec_lf = encoded_st_lf
                else:
                    dec_lf = [UNK_TOKEN for _ in range(args.max_input_size)]
                    dec_lf_tensor = self._prepare_input_tensor(dec_lf, self.vocabs[LOGICAL_FORM])
                    encoded_dec_lf = self.model.query_encoder(dec_lf_tensor)

            cross_attn = self.model.cross_attention(encoded_question, encoded_dec_lf)

            verbalization = [self.vocabs[DECODER].stoi[START_TOKEN]]

            for _ in range(self.model.decoder.max_positions):
                verbalization_tensor = torch.LongTensor(verbalization).unsqueeze(0).to(self.device)

                decoder_out, _ = self.model.decoder(question_tensor, verbalization_tensor, cross_attn)

                predicted_token = decoder_out.argmax(1)[-1].item()

                if predicted_token == self.vocabs[DECODER].stoi[END_TOKEN]:
                    break

                verbalization.append(predicted_token)

        return {
            DECODER: [self.vocabs[DECODER].itos[i] for i in verbalization][1:],
            SIMILARITY_THRESHOLD: [self.vocabs[SIMILARITY_THRESHOLD].itos[i] for i in st_pred]
        }

class Scorer(object):
    def __init__(self):
        self.results = []
        self.bleu_score_meter = AverageMeter()
        self.meteor_score_meter = AverageMeter()
        self.similarity_threshold_meter = F1ScoreMeter()

    def _bleu_score(self, reference, hypothesis):
        return nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)

    def _meteor_score(self, reference, hypothesis):
        return nltk.translate.meteor_score.single_meteor_score(' '.join(reference), ' '.join(hypothesis))

    def _hash_text(self, text):
        return int(hashlib.sha1(text.encode("utf-8")).hexdigest(), 16) % (10 ** 8)

    def data_score(self, data, predictor):
        seen_examples = {}
        for example in tqdm.tqdm(data):
            reference = [t.lower() for t in example.decoder]
            hypothesis = predictor.predict(example.input, example.st_logical_form)

            blue_score = self._bleu_score(reference, hypothesis[DECODER])
            meteor_score = self._meteor_score(reference, hypothesis[DECODER])

            self.results.append({
                REFERENCE: hypothesis[DECODER],
                HYPOTHESIS: hypothesis,
                BLEU_SCORE: blue_score,
                METEOR_SCORE: blue_score
            })

            if args.dataset == PARAQA: # for ParaQA get max values (BLEU, METEOR)
                input_hash = self._hash_text(''.join(example.input) + example.similarity_threshold[0])
                if input_hash in seen_examples:
                    seen_examples[input_hash][BLEU_SCORE].append(blue_score)
                    seen_examples[input_hash][METEOR_SCORE].append(meteor_score)
                else:
                    seen_examples[input_hash] = {
                        BLEU_SCORE: [blue_score],
                        METEOR_SCORE: [meteor_score]
                    }
            else:
                self.bleu_score_meter.update(blue_score)
                self.meteor_score_meter.update(meteor_score)

            self.similarity_threshold_meter.update(int(example.similarity_threshold[0]), int(hypothesis[SIMILARITY_THRESHOLD][0]))

        if args.dataset == PARAQA:
            for para_res in seen_examples.values():
                self.bleu_score_meter.update(max(para_res[BLEU_SCORE]))
                self.meteor_score_meter.update(max(para_res[METEOR_SCORE]))

        return {
            RESULTS: self.results,
            BLEU_SCORE: self.bleu_score_meter.avg,
            METEOR_SCORE: self.meteor_score_meter.avg,
            ST_SCORE: self.similarity_threshold_meter.f1_score
        }

    def reset(self):
        self.results = []
        self.score_meter = AverageMeter()

class AccuracyMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.wrong = 0
        self.accuracy = 0

    def update(self, gold, result):
        if gold == result:
            self.correct += 1
        else:
            self.wrong += 1

        self.accuracy = self.correct / (self.correct + self.wrong)

class SingleTaskLoss(nn.Module):
    def __init__(self, ignore_index):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, output, target):
        return self.criterion(output, target)

class MultiTaskLoss(nn.Module):
    def __init__(self, ignore_index):
        super().__init__()
        self.decoder_loss = SingleTaskLoss(ignore_index)
        self.similarity_threshold_loss = SingleTaskLoss(ignore_index)

        self.mml_emp = torch.Tensor([True, True])
        self.log_vars = torch.nn.Parameter(torch.zeros(len(self.mml_emp)))

    def forward(self, output, target):
        task_losses = torch.stack((
            self.decoder_loss(output[DECODER], target[DECODER]),
            self.similarity_threshold_loss(output[SIMILARITY_THRESHOLD], target[SIMILARITY_THRESHOLD]),
        ))

        dtype = task_losses.dtype
        stds = (torch.exp(self.log_vars)**(1/2)).to(DEVICE).to(dtype)
        weights = 1 / ((self.mml_emp.to(DEVICE).to(dtype)+1)*(stds**2))

        losses = weights * task_losses + torch.log(stds)

        return {
            DECODER: losses[0],
            SIMILARITY_THRESHOLD: losses[1],
            MULTITASK: losses.mean()
        }[args.task]

def evaluate(loader, model, vocabs, criterion):
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for _, batch in enumerate(loader):
            # get inputs
            input = batch.input
            logical_forms = {
                ST_LOGICAL_FORM: batch.st_logical_form,
                DEC_LOGICAL_FORM: batch.dec_logical_form
            }
            similarity_threshold = batch.similarity_threshold
            decoder = batch.decoder

            # compute output
            output = model(input, logical_forms, decoder[:, :-1])

            # prepare targets
            target = {
                DECODER: decoder[:, 1:].contiguous().view(-1), # (batch_size * trg_len)
                SIMILARITY_THRESHOLD: similarity_threshold[:, 1:2].contiguous().view(-1) # exclude [start], [end] tokens
            }

            # compute loss
            loss = criterion(output, target) if args.task == MULTITASK else criterion(output[args.task], target[args.task])

            # record loss
            losses.update(loss.data, input.size(0))

    return losses.avg

def init_weights(model):
    # initialize model parameters with Glorot / fan_avg
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

def save_checkpoint(state):
    filename = f'{ROOT_PATH}/{args.snapshots}/{args.dataset}_e{state[EPOCH]}_v{state[CURR_VAL]:.4f}_{args.task}.pth.tar'
    torch.save(state, filename)
