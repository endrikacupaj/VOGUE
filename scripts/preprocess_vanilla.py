import re
import json
import tqdm
import random
from flair.data import Sentence
from flair.models import SequenceTagger

from constants import *

# load the NER tagger
tagger = SequenceTagger.load('ner')

def cover_answer(sentence, answer):
    return sentence.replace(answer, ANS_TOKEN)

def prepare_query(predicate):
    return {
        LOGICAL_FORM: f'find {ENT_TOKEN.lower()} {predicate.lower()}'
    }

def create_negative_examples(true_predicate, predicates):
    random_predicate = random.choice(predicates)
    while random_predicate == true_predicate:
        random_predicate = random.choice(predicates)

    return f'find {ENT_TOKEN.lower()} {random_predicate.lower()}'

def cover_entities(text, answer=None, is_answer=False):
    if is_answer and answer:
        text = cover_answer(text, answer)

    sentence = Sentence(text)
    tagger.predict(sentence)

    for entity in sentence.get_spans('ner'):
        if entity.text != 'ANS':
            text = text.replace(entity.text, ENT_TOKEN)

    return text

def preprocess(data, all_predicates):
    for d in tqdm.tqdm(data):
        uid = d['question_id']
        question = d['question']
        answer = d['answer'].lower()
        answer_sentence = d['answer_sentence']
        predicate = d['question_relation']

        # cover entities
        covered_entities_question = cover_entities(question)
        covered_entities_answer = cover_entities(answer_sentence, answer, is_answer=True)

        # preprocess queries
        preprocessed_queries = prepare_query(predicate)

        # negative examples
        negative_example = create_negative_examples(predicate, all_predicates)

        d[PREPROCESSED] = {
            QUESTION: covered_entities_question,
            ANSWER: covered_entities_answer,
            LOGICAL_FORM: preprocessed_queries[LOGICAL_FORM],
            NEGATIVE: negative_example
        }

    return data

def write_data(data, name):
    with open(f'{str(ROOT_PATH)}/data/vanilla/preprocessed_{name}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    train_data = json.load(open(f'{str(ROOT_PATH)}/data/vanilla/train.json'))
    test_data = json.load(open(f'{str(ROOT_PATH)}/data/vanilla/test.json'))

    predicates = list(set([data['question_relation'] for data in train_data + test_data]))

    train_data = preprocess(train_data, predicates)
    write_data(train_data, 'train')

    test_data = preprocess(test_data, predicates)
    write_data(test_data, 'test')
