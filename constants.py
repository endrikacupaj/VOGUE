import os
import torch
from pathlib import Path
from args import get_parser

# set root path
ROOT_PATH = Path(os.path.dirname(__file__))

# read parser
parser = get_parser()
args = parser.parse_args()

# model name
MODEL_NAME = 'VOGUE'

# define device
CUDA = 'cuda'
CPU = 'cpu'
DEVICE = torch.device(CUDA if torch.cuda.is_available() else CPU)

# datasets
VQUANDA = 'vquanda'
PARAQA = 'paraqa'
VANILLA = 'vanilla'

# fields
INPUT = 'input'
DECODER = 'decoder'
LOGICAL_FORM = 'logical_form'
ST_LOGICAL_FORM = 'st_logical_form'
DEC_LOGICAL_FORM = 'dec_logical_form'
QUERY = 'query'
SIMILARITY_THRESHOLD = 'similarity_threshold'
MULTITASK = 'multitask'

# helper tokens
START_TOKEN = '[START]'
END_TOKEN = '[END]'
CTX_TOKEN = '[CTX]'
PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
NA_TOKEN = 'NA'
ENT_TOKEN = '[ENT]'
ANS_TOKEN = '[ANS]'

# model
ENCODER_OUT = 'encoder_out'
DECODER_OUT = 'decoder_out'

# training
EPOCH = 'epoch'
STATE_DICT = 'state_dict'
BEST_VAL = 'best_val'
OPTIMIZER = 'optimizer'
CURR_VAL = 'curr_val'

# preprocessing
CLASS = 'class'
QUERY_TYPE = 'query_type'
QUERY_PREDICATES = 'query_predicates'
NEGATIVE = 'negative'
PREPROCESSED = 'preprocessed'

# other
QUESTION = 'question'
ANSWER = 'answer'
RESULTS = 'results'
REFERENCE = 'reference'
HYPOTHESIS = 'hypothesis'
BLEU_SCORE = 'bleu_score'
METEOR_SCORE = 'meteor_score'
ST_SCORE = 'st_score'
ENTITY = 'entity'
RELATION = 'relation'
TYPE = 'type'