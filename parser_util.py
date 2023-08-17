# coding=utf-8
import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root',
                        type=str,
                        help='path to dataset',
                        default='..' + os.sep + 'dataset')

    parser.add_argument('--experiment_root',
                        type=str,
                        help='root where to store models, losses and accuracies',
                        default='..' + os.sep + 'output')

    parser.add_argument('--epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=100)

    parser.add_argument('--startIter',
                        type=int,
                        help='Start iteration',
                        default=0)


    parser.add_argument('--maxIter',
                        type=int,
                        help='Max iteration',
                        default=100000)
    
    parser.add_argument('--valIter',
                        type=int,
                        help='Validation iteration',
                        default=100)


    parser.add_argument('--learning_rate',
                        type=float,
                        help='learning rate for the model, default=2.0',
                        default=2.0)

    parser.add_argument('--rho',
                            type=float,
                            help='rho parameter in AdaAdam, default=0.9',
                            default=2.0)

    parser.add_argument('--eps',
                            type=float,
                            help='eps parameter in AdaAdam, default=1e-05',
                            default=1e-05)
    parser.add_argument('--seed',
                        type=int,
                        help='input for the manual seeds initializations',
                        default=2103)

    parser.add_argument('--test',
                        action='store_true',
                        help='Test mode')
    
    parser.add_argument('--train',
                        action='store_true',
                        help='Train mode')




    return parser
