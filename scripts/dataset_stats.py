import re
import json
import tqdm
from torchtext.data import Field, Example, Dataset

from constants import *

TOKENIZE_SEQ = lambda x: x.replace("?", " ?").\
                                     replace(".", " .").\
                                     replace("[", "[ ").\
                                     replace("]", " ]").\
                                     replace(",", " ,").\
                                     replace("'", " '").\
                                     split()

def make_torchtext_dataset(data, fields):
    examples = [Example.fromlist(i, fields) for i in data]
    return Dataset(examples, fields)

# VQUANDA
vquanda_dataset = json.load(open(f'{ROOT_PATH}/data/vquanda/train.json')) + json.load(open(f'{ROOT_PATH}/data/vquanda/test.json'))
vquanda_questions = [TOKENIZE_SEQ(v['question'].lower()) for v in vquanda_dataset]
vquanda_answers = [TOKENIZE_SEQ(v['verbalized_answer'].lower()) for v in vquanda_dataset]

# Avg tokens per sentence
print(sum([len(v) for v in vquanda_questions])/len(vquanda_questions))
print(sum([len(v) for v in vquanda_answers])/len(vquanda_answers))

# vocabulary
vquanda_input_field = Field(init_token=START_TOKEN,
                    eos_token=CTX_TOKEN,
                    pad_token=PAD_TOKEN,
                    unk_token=UNK_TOKEN,
                    lower=True,
                    batch_first=True)
fields_tuple = [(INPUT, vquanda_input_field)]

vquadna_vocab_data = make_torchtext_dataset([[d] for d in vquanda_questions + vquanda_answers], fields_tuple)
vquanda_input_field.build_vocab(vquadna_vocab_data, min_freq=0, vectors='glove.840B.300d')

print(len(vquanda_input_field.vocab))

# ParaQA
paraqa_dataset = json.load(open(f'{ROOT_PATH}/data/paraqa/train.json')) + json.load(open(f'{ROOT_PATH}/data/paraqa/test.json'))
paraqa_questions = [TOKENIZE_SEQ(v['question'].lower()) for v in paraqa_dataset]
paraqa_answers = [[TOKENIZE_SEQ(v['verbalized_answer'].lower()), TOKENIZE_SEQ(v['verbalized_answer_2'].lower()), TOKENIZE_SEQ(v['verbalized_answer_3'].lower()), TOKENIZE_SEQ(v['verbalized_answer_4'].lower()), TOKENIZE_SEQ(v['verbalized_answer_5'].lower()), TOKENIZE_SEQ(v['verbalized_answer_6'].lower()), TOKENIZE_SEQ(v['verbalized_answer_7'].lower()), TOKENIZE_SEQ(v['verbalized_answer_8'].lower())] for v in paraqa_dataset]
paraqa_answers = [item for sublist in paraqa_answers for item in sublist if len(item) > 0]

# Avg tokens per sentence
print(sum([len(v) for v in paraqa_questions])/len(paraqa_questions))
print(sum([len(v) for v in paraqa_answers])/len(paraqa_answers))

# vocabulary
paraqa_input_field = Field(init_token=START_TOKEN,
                    eos_token=CTX_TOKEN,
                    pad_token=PAD_TOKEN,
                    unk_token=UNK_TOKEN,
                    lower=True,
                    batch_first=True)
fields_tuple = [(INPUT, paraqa_input_field)]

paraqa_vocab_data = make_torchtext_dataset([[d] for d in paraqa_questions + paraqa_answers], fields_tuple)
paraqa_input_field.build_vocab(paraqa_vocab_data, min_freq=0, vectors='glove.840B.300d')

print(len(paraqa_input_field.vocab))

# VANILLA
vanilla_dataset = json.load(open(f'{ROOT_PATH}/data/vanilla/train.json')) + json.load(open(f'{ROOT_PATH}/data/vanilla/test.json'))
vanilla_questions = [TOKENIZE_SEQ(v['question'].lower()) for v in vanilla_dataset]
vanilla_answers = [TOKENIZE_SEQ(v['answer_sentence'].lower()) for v in vanilla_dataset]

# Avg tokens per sentence
print(sum([len(v) for v in vanilla_questions])/len(vanilla_questions))
print(sum([len(v) for v in vanilla_answers])/len(vanilla_answers))

# vocabulary
vanilla_input_field = Field(init_token=START_TOKEN,
                    eos_token=CTX_TOKEN,
                    pad_token=PAD_TOKEN,
                    unk_token=UNK_TOKEN,
                    lower=True,
                    batch_first=True)
fields_tuple = [(INPUT, vanilla_input_field)]

vanilla_vocab_data = make_torchtext_dataset([[d] for d in vanilla_questions + vanilla_answers], fields_tuple)
vanilla_input_field.build_vocab(vanilla_vocab_data, min_freq=0, vectors='glove.840B.300d')

print(len(vanilla_input_field.vocab))