import time
import torch
import random
import logging
import numpy as np
from model import VOGUE
from args import get_parser
from torchtext.data import BucketIterator
from data.dataset import AnswerVerbalizationDataset

from constants import *
from utils import (NoamOpt, AverageMeter,
                    SingleTaskLoss, MultiTaskLoss,
                    save_checkpoint, init_weights, evaluate)

# set logger
logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler(f'{args.path_results}/train_{args.dataset}_{args.task}.log', 'w'),
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
    train_data, val_data, _ = dataset.get_data()

    # load model
    model = VOGUE(vocabs).to(DEVICE)

    # initialize model weights
    init_weights(model)

    logger.info(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

    # define loss function (criterion)
    criterion = {
        DECODER: SingleTaskLoss,
        SIMILARITY_THRESHOLD: SingleTaskLoss,
        MULTITASK: MultiTaskLoss
    }[args.task](ignore_index=vocabs[DECODER].stoi[PAD_TOKEN])

    # define optimizer
    optimizer = NoamOpt(torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"=> loading checkpoint '{args.resume}''")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint[EPOCH]
            best_val = checkpoint[BEST_VAL]
            model.load_state_dict(checkpoint[STATE_DICT])
            optimizer.optimizer.load_state_dict(checkpoint[OPTIMIZER])
            logger.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint[EPOCH]})")
        else:
            logger.info(f"=> no checkpoint found at '{args.resume}'")
            best_val = float('inf')
    else:
        best_val = float('inf')

    # prepare training and validation loader
    train_loader, val_loader = BucketIterator.splits((train_data, val_data),
                                                    batch_size=args.batch_size,
                                                    sort_within_batch=False,
                                                    sort_key=lambda x: len(x.input),
                                                    device=DEVICE)

    logger.info('Loaders prepared')
    logger.info(f'Training on {args.dataset} dataset')
    logger.info(f'Training data: {len(train_data.examples)}')
    logger.info(f'Validation data: {len(val_data.examples)}')
    logger.info(f'Question example: {train_data.examples[0].input}')
    logger.info(f'Logical form example: {train_data.examples[0].decoder}')
    logger.info(f'Unique tokens in input vocabulary: {len(vocabs[INPUT])}')
    logger.info(f'Unique tokens in decoder vocabulary: {len(vocabs[DECODER])}')
    logger.info(f'Batch: {args.batch_size}')
    logger.info(f'Epochs: {args.epochs}')

    # run epochs
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, vocabs, criterion, optimizer, epoch)

        # evaluate on validation set
        if (epoch+1) % args.valfreq == 0:
            val_loss = evaluate(val_loader, model, vocabs, criterion)
            if val_loss < best_val:
                best_val = min(val_loss, best_val)
                save_checkpoint({
                    EPOCH: epoch + 1,
                    STATE_DICT: model.state_dict(),
                    BEST_VAL: best_val,
                    OPTIMIZER: optimizer.optimizer.state_dict(),
                    CURR_VAL: val_loss})
            logger.info(f'* Val loss: {val_loss:.4f}')

def train(train_loader, model, vocabs, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, batch in enumerate(train_loader):
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

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        logger.info(f'Epoch: {epoch+1} - Train loss: {losses.val:.4f} ({losses.avg:.4f}) - Batch: {((i+1)/len(train_loader))*100:.2f}% - Time: {batch_time.sum:0.2f}s')

if __name__ == '__main__':
    main()