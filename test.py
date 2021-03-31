import time
import torch
import random
import logging
import numpy as np
from args import get_parser
from torchtext.data import BucketIterator
from model import HybridAnswerVerbalization
from data.dataset import AnswerVerbalizationDataset

from constants import *
from utils import (SingleTaskLoss, MultiTaskLoss,
                    Scorer, Predictor, evaluate)

# set logger
logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler(f'{args.path_results}/test_{args.dataset}_{args.task}.log', 'w'),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# set a seed value
random.seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

# set cuda device
torch.cuda.set_device(args.cuda_device)

def main():
    # load data
    dataset = AnswerVerbalizationDataset()
    vocabs = dataset.get_vocabs()
    _, val_data, test_data = dataset.get_data()

    # load model
    model = HybridAnswerVerbalization(vocabs).to(DEVICE)

    # define loss function (criterion)
    criterion = {
        DECODER: SingleTaskLoss,
        SIMILARITY_THRESHOLD: SingleTaskLoss,
        MULTITASK: MultiTaskLoss
    }[args.task](ignore_index=vocabs[DECODER].stoi[PAD_TOKEN])

    logger.info(f"=> loading checkpoint '{args.model_path}'")
    if DEVICE.type==CPU:
        checkpoint = torch.load(f'{ROOT_PATH}/{args.model_path}', encoding='latin1', map_location=CPU)
    else:
        checkpoint = torch.load(f'{ROOT_PATH}/{args.model_path}', encoding='latin1')
    args.start_epoch = checkpoint[EPOCH]
    model.load_state_dict(checkpoint[STATE_DICT])
    logger.info(f"=> loaded checkpoint '{args.model_path}' (epoch {checkpoint[EPOCH]})")

    # prepare validation and test loader
    val_loader, test_loader = BucketIterator.splits((val_data, test_data),
                                                    batch_size=args.batch_size,
                                                    sort_within_batch=False,
                                                    sort_key=lambda x: len(x.input),
                                                    device=DEVICE)

    logger.info('Loaders prepared')
    logger.info(f'Testing on {args.dataset} dataset')
    logger.info(f"Validation data: {len(val_data.examples)}")
    logger.info(f"Test data: {len(test_data.examples)}")

    # calculate loss
    val_loss = evaluate(val_loader, model, vocabs, criterion)
    logger.info(f'* Val Loss: {val_loss:.4f}')
    test_loss = evaluate(test_loader, model, vocabs, criterion)
    logger.info(f'* Test Loss: {test_loss:.4f}')

    # predict results
    predictor = Predictor(model, vocabs, DEVICE)
    scorer = Scorer()

    scorer.data_score(val_data, predictor)
    scorer.write_results()
    logger.info(f'Similarity Threshold Accuracy: {scorer.similarity_threshold_meter.f1_score}')
    logger.info(f'BLEU score: {scorer.bleu_score_meter.avg}')
    logger.info(f'METEOR score: {scorer.meteor_score_meter.avg}')

    scorer.data_score(test_data, predictor)
    scorer.write_results()
    logger.info(f'Similarity Threshold Accuracy: {scorer.similarity_threshold_meter.f1_score}')
    logger.info(f'BLEU score: {scorer.bleu_score_meter.avg}')
    logger.info(f'METEOR score: {scorer.meteor_score_meter.avg}')

if __name__ == '__main__':
    main()