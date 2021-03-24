import re
import json
import tqdm
from fuzzywuzzy import fuzz
from flair.data import Sentence
from flair.models import SequenceTagger

from constants import *

ANSWER_REGEX = r'\[.*?\]'
QUERY_DICT = {
    'x': 'var_x',
    'uri': 'var_uri',
    '{': 'brack_open',
    '}': 'brack_close',
    '.': 'sep_dot',
    'COUNT(uri)': 'count_uri'
}

# load the NER tagger
tagger = SequenceTagger.load('ner')

def read_sparql_template_ids():
    lcquad_data = json.load(open(str(ROOT_PATH) + args.data_path + '/lcquad/train.json'))
    lcquad_data.extend(json.load(open(str(ROOT_PATH) + args.data_path + '/lcquad/test.json')))

    return { data['_id']: data['sparql_template_id'] for data in lcquad_data }

def cover_answer(text):
    return re.sub(ANSWER_REGEX, ANS_TOKEN, text)

def prepare_query(query, uid, cover_entities=True):
    sparql_template_id = sparql_templates_ids[uid]
    template = templates[sparql_template_id]

    logical_form = template[LOGICAL_FORM]
    query_type = template[TYPE]

    logical_form = logical_form.replace('(', ' ( ')\
                                .replace(')', ' ) ')\
                                .replace(',', ' ,')\
                                .replace('ent', ENT_TOKEN.lower()).strip()

    query = query.replace('\n', ' ')\
                    .replace('\t', '')\
                    .replace('?', '')\
                    .replace('{?', '{ ?')\
                    .replace('>}', '> }')\
                    .replace('{uri', '{ uri')\
                    .replace('uri}', 'uri }').strip().split()

    predicates = [
        predicate.replace(',\n', '')
        for predicate in open(str(ROOT_PATH) + args.data_path + '/predicates.txt').readlines()
    ]

    predicate_counter = 1
    query_predicates = []
    has_query_type = 0
    query_type = None
    new_query = []

    for q in query:
        if q in QUERY_DICT:
            q = QUERY_DICT[q]
        if 'http' in q:
            if 'dbpedia.org/ontology' in q or 'dbpedia.org/property' in q:
                uri = q.replace('<', '').replace('>', '')
                q = q.rsplit('/', 1)[-1].lstrip('<').rstrip('>').lower()

                if uri in predicates:
                    logical_form = logical_form.replace(f'pred{predicate_counter}', q)
                    predicate_counter += 1
                    query_predicates.append(q)

                if has_query_type:
                    logical_form = logical_form.replace('class', q)
                    query_type = q
                    has_query_type = False
            elif 'www.w3.org/1999/02/22-rdf-syntax-ns#type' in q:
                q = 'type'
                has_query_type = True
            elif cover_entities:
                q = ENT_TOKEN
            else:
                q = q.rsplit('/', 1)[-1].lstrip('<').rstrip('>') # .replace('_', ' ')

        new_query.append(q.lower())

    assert new_query[-1] == "brack_close", "Query not ending with a bracket."

    return {
        QUERY: ' '.join(new_query),
        LOGICAL_FORM: logical_form,
        QUERY_TYPE: query_type,
        QUERY_PREDICATES: query_predicates
    }

def find_similar_negative(template_id, k=2):
    template = templates[template_id]
    logical_form = template[LOGICAL_FORM]

    similarity_temp = {
        temp['id']: fuzz.ratio(logical_form, temp[LOGICAL_FORM]) for temp in list(templates.values()) if temp['id'] != template['id'] and temp[TYPE] == template[TYPE]
    }

    sorted_similar_templates = [sim[0] for sim in sorted(similarity_temp.items(), key=lambda item: item[1], reverse=True) if sim != 100]

    return sorted_similar_templates[:k]

def create_negative_examples(uid, predicates, query_type):
    sparql_template_id = sparql_templates_ids[uid]
    negative_ids = find_similar_negative(sparql_template_id)

    negative_examples = []
    for id in negative_ids:
        template = templates[id]
        logical_form = template[LOGICAL_FORM]

        logical_form = logical_form.replace('(', ' ( ')\
                                   .replace(')', ' ) ')\
                                   .replace(',', ' ,')\
                                   .replace('ent', ENT_TOKEN.lower()).strip()

        # replace type
        if query_type is not None:
            logical_form = logical_form.replace(CLASS, query_type)

        # replace predicates
        for i, predicate in enumerate(predicates):
            logical_form = logical_form.replace(f'pred{str(i+1)}', predicate)

        negative_examples.append(logical_form)

    return negative_examples

def cover_entities(text, is_answer=False):
    if is_answer:
        text = cover_answer(text)

    sentence = Sentence(text)
    tagger.predict(sentence)

    for entity in sentence.get_spans('ner'):
        if entity.text != 'ANS':
            text = text.replace(entity.text, ENT_TOKEN)

    return text

def preprocess(data):
    for d in tqdm.tqdm(data):
        uid = d['uid']
        question = d['question']
        answer = d['verbalized_answer']
        query = d['query']

        # cover entities
        covered_entities_question = cover_entities(question)
        covered_entities_answer = cover_entities(answer, is_answer=True)

        # preprocess queries
        preprocessed_queries = prepare_query(query, uid)

        # negative examples
        negative_examples = create_negative_examples(uid, preprocessed_queries[QUERY_PREDICATES], preprocessed_queries[QUERY_TYPE])

        d[PREPROCESSED] = {
            QUESTION: covered_entities_question,
            ANSWER: covered_entities_answer,
            QUERY: preprocessed_queries[QUERY],
            LOGICAL_FORM: preprocessed_queries[LOGICAL_FORM],
            NEGATIVE: negative_examples
        }

    return data

def write_data(data, name):
    with open(f'{str(ROOT_PATH)}/data/vquanda/preprocessed_{name}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    templates = { temp['id']: temp for temp in json.load(open(str(ROOT_PATH) + args.data_path + '/templates.json')) }
    sparql_templates_ids = read_sparql_template_ids()

    train_data = json.load(open(f'{str(ROOT_PATH)}/data/vquanda/train.json'))
    test_data = json.load(open(f'{str(ROOT_PATH)}/data/vquanda/test.json'))

    train_data = preprocess(train_data)
    test_data = preprocess(test_data)

    write_data(train_data, 'train')
    write_data(test_data, 'test')
