import os
import re
import json
import tqdm
import spacy
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchtext.data import Field, Example, Dataset

# import constants
from constants import *

class AnswerVerbalizationDataset(object):
    TOKENIZE_SEQ = lambda self, x: x.replace("?", " ?").\
                                     replace(".", " .").\
                                     replace(",", " ,").\
                                     replace("'", " '").\
                                     split()

    def __init__(self):
        self.train_path = str(ROOT_PATH) + '/data/' + args.dataset + '/preprocessed_train.json'
        self.test_path = str(ROOT_PATH) + '/data/' + args.dataset + '/preprocessed_test.json'
        self.prepare_data = {
            VQUANDA: self._prepare_vquanda,
            PARAQA: self._prepare_paraqa,
            VANILLA: self._prepare_vanilla
        }
        self.load_data_and_fields()

    def _prepare_vquanda(self, data):
        input_data = []
        for d in data:
            uid = d['uid']
            preprrocessed = d[PREPROCESSED]

            # preprocess
            question = preprrocessed[QUESTION].lower()
            logical_form = preprrocessed[LOGICAL_FORM].lower()
            answer_verbalization = preprrocessed[ANSWER].lower()
            negative_lf = preprrocessed[NEGATIVE]

            # tokenize
            input = self.TOKENIZE_SEQ(question)
            logical_form = self.TOKENIZE_SEQ(logical_form)
            answer_verbalization = self.TOKENIZE_SEQ(answer_verbalization)

            while len(input) < args.max_input_size:
                input.append(PAD_TOKEN)

            while len(logical_form) < args.max_input_size:
                logical_form.append(PAD_TOKEN)

            input_data.append([input, logical_form, logical_form, ['1'], answer_verbalization])

            for negative in negative_lf[:1]:
                negative = self.TOKENIZE_SEQ(negative.lower())
                dec_logical_form = [PAD_TOKEN for _ in range(args.max_input_size)]

                input_data.append([input, negative, dec_logical_form, ['0'], answer_verbalization])

        return input_data

    def _prepare_paraqa(self, data):
        input_data = []
        for d in data:
            uid = d['uid']
            preprrocessed = d[PREPROCESSED]

            question = preprrocessed[QUESTION].lower()
            logical_form = preprrocessed[LOGICAL_FORM].lower()
            answers = preprrocessed[ANSWER]
            negative_lf = preprrocessed[NEGATIVE]

            input = self.TOKENIZE_SEQ(question)
            logical_form = self.TOKENIZE_SEQ(logical_form)

            while len(input) < args.max_input_size:
                input.append(PAD_TOKEN)

            while len(logical_form) < args.max_input_size:
                    logical_form.append(PAD_TOKEN)

            for answer in answers:
                answer_verbalization = self.TOKENIZE_SEQ(answer.lower())

                input_data.append([input, logical_form, logical_form, ['1'], answer_verbalization])

                for negative in negative_lf:
                    negative = self.TOKENIZE_SEQ(negative.lower())
                    dec_logical_form = [UNK_TOKEN for _ in range(args.max_input_size)]

                    input_data.append([input, negative, dec_logical_form, ['0'], answer_verbalization])

        return input_data

    def _prepare_vanilla(self, data):
        input_data = []
        for d in data:
            uid = d['question_id']
            preprrocessed = d[PREPROCESSED]

            question = preprrocessed[QUESTION].lower()
            logical_form = preprrocessed[LOGICAL_FORM].lower()
            answer_verbalization = preprrocessed[ANSWER].lower()
            negative_lf = preprrocessed[NEGATIVE]

            input = self.TOKENIZE_SEQ(question)
            logical_form = self.TOKENIZE_SEQ(logical_form)
            answer_verbalization = self.TOKENIZE_SEQ(answer_verbalization)
            negative_lf = self.TOKENIZE_SEQ(negative_lf)

            if len(input) > args.max_input_size or\
               len(logical_form) > args.max_input_size or\
               len(negative_lf) > args.max_input_size:
                continue

            while len(input) < args.max_input_size:
                input.append(PAD_TOKEN)

            while len(logical_form) < args.max_input_size:
                logical_form.append(PAD_TOKEN)

            input_data.append([input, logical_form, logical_form, ['1'], answer_verbalization])

            while len(negative_lf) < args.max_input_size:
                negative_lf.append(PAD_TOKEN)

            dec_logical_form = [PAD_TOKEN for _ in range(args.max_input_size)]

            input_data.append([input, negative_lf, dec_logical_form, ['0'], answer_verbalization])

        return input_data

    def _make_torchtext_dataset(self, data, fields):
        examples = [Example.fromlist(i, fields) for i in data]
        return Dataset(examples, fields)

    def load_data_and_fields(self, cover_entities=False, query_as_input=False):
        train, test, val = [], [], []
        # read data
        with open(self.train_path) as json_file:
            train = json.load(json_file)

        with open(self.test_path) as json_file:
            test = json.load(json_file)

        test, val = train_test_split(test, test_size=0.4, shuffle=False)

        train = self.prepare_data[args.dataset](train)
        val = self.prepare_data[args.dataset](val)
        test = self.prepare_data[args.dataset](test)

        # create fields
        self.input_field = Field(init_token=START_TOKEN,
                                eos_token=CTX_TOKEN,
                                pad_token=PAD_TOKEN,
                                unk_token=UNK_TOKEN,
                                lower=True,
                                batch_first=True)

        self.lf_field = Field(init_token=START_TOKEN,
                                eos_token=CTX_TOKEN,
                                pad_token=PAD_TOKEN,
                                unk_token=UNK_TOKEN,
                                lower=True,
                                batch_first=True)

        self.sim_field = Field(init_token='0',
                                eos_token='0',
                                pad_token=PAD_TOKEN,
                                unk_token='0',
                                batch_first=True)

        self.decoder_field = Field(init_token=START_TOKEN,
                                eos_token=END_TOKEN,
                                pad_token=PAD_TOKEN,
                                unk_token=UNK_TOKEN,
                                lower=True,
                                batch_first=True)

        fields_tuple = [(INPUT, self.input_field),
                        (ST_LOGICAL_FORM, self.lf_field),
                        (DEC_LOGICAL_FORM, self.lf_field),
                        (SIMILARITY_THRESHOLD, self.sim_field),
                        (DECODER, self.decoder_field)]

        # create toechtext datasets
        self.train_data = self._make_torchtext_dataset(train, fields_tuple)
        self.val_data = self._make_torchtext_dataset(val, fields_tuple)
        self.test_data = self._make_torchtext_dataset(test, fields_tuple)

        # build vocabularies
        self.input_field.build_vocab(self.train_data, self.val_data, self.test_data, min_freq=0, vectors='glove.840B.300d')
        self.lf_field.build_vocab(self.train_data, self.val_data, self.test_data, min_freq=0, vectors='glove.840B.300d')
        self.sim_field.build_vocab(self.train_data, self.val_data, self.test_data, min_freq=0)
        self.decoder_field.build_vocab(self.train_data, self.val_data, self.test_data, min_freq=0)

    def get_data(self):
        return self.train_data, self.val_data, self.test_data

    def get_fields(self):
        return {
            INPUT: self.input_field,
            LOGICAL_FORM: self.lf_field,
            SIMILARITY_THRESHOLD: self.sim_field,
            DECODER: self.decoder_field,
        }

    def get_vocabs(self):
        return {
            INPUT: self.input_field.vocab,
            LOGICAL_FORM: self.lf_field.vocab,
            SIMILARITY_THRESHOLD: self.sim_field.vocab,
            DECODER: self.decoder_field.vocab
        }
