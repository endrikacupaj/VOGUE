import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='VOGUE: Answer Verbalization through Multi Task Learning')

    # general
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--cuda_device', default=0, type=int)

    # dataset and task
    parser.add_argument('--dataset', default='vquanda', choices=['vquanda', 'paraqa', 'vanilla'], type=str)
    parser.add_argument('--task', default='multitask', choices=['multitask',
                                                                'similarity_threshold',
                                                                'decoder'], type=str)

    # model
    parser.add_argument('--emb_dim', default=300, type=int)
    parser.add_argument('--dropout', default=0.1, type=int)
    parser.add_argument('--heads', default=6, type=int)
    parser.add_argument('--layers', default=2, type=int)
    parser.add_argument('--max_positions', default=1000, type=int)
    parser.add_argument('--max_input_size', default=30, type=int)
    parser.add_argument('--pf_dim', default=300, type=int)

    # training
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--warmup', default=4000, type=float)
    parser.add_argument('--factor', default=1, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--valfreq', default=1, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--clip', default=5, type=int)
    parser.add_argument('--batch_size', default=256, type=int)

    # test and experiments
    parser.add_argument('--snapshots', default='experiments/snapshots', type=str)
    parser.add_argument('--path_results', default='experiments/results', type=str)
    parser.add_argument('--model_path', default='experiments/snapshots/', type=str)

    return parser